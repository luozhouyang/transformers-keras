import json
import logging
import os

import tensorflow as tf

from transformers_keras.adapters import parse_pretrained_model_files
from transformers_keras.adapters.albert_adapter import AlbertAdapter

from .modeling_bert import BertEmbedding, BertEncoderLayer, BertIntermediate
from .modeling_utils import choose_activation, unpack_inputs_3


class AlbertEmbedding(tf.keras.layers.Layer):

    def __init__(self,
                 vocab_size=21128,
                 max_positions=512,
                 type_vocab_size=2,
                 embedding_size=128,
                 hidden_dropout_rate=0.2,
                 initializer_range=0.02,
                 epsilon=1e-8,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_positions = max_positions
        self.type_vocab_size = type_vocab_size
        self.embedding_size = embedding_size
        self.initializer_range = initializer_range

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=epsilon, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(hidden_dropout_rate)

    def build(self, input_shape):
        self.word_embedding = self.add_weight(
            name='word_embeddings',
            shape=[self.vocab_size, self.embedding_size],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range))
        self.position_embedding = self.add_weight(
            name='position_embeddings',
            shape=[self.max_positions, self.embedding_size],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range))
        self.segment_embedding = self.add_weight(
            name='token_type_embeddings',
            shape=[self.type_vocab_size, self.embedding_size],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range))
        super().build(input_shape)

    def call(self, input_ids, segment_ids=None, position_ids=None, training=None):
        if segment_ids is None:
            segment_ids = tf.zeros_like(input_ids)
        if position_ids is None:
            position_ids = tf.range(0, tf.shape(input_ids)[1], dtype=input_ids.dtype)
            position_ids = tf.expand_dims(position_ids, axis=0)

        position_embeddings = tf.gather(self.position_embedding, position_ids)
        position_embeddings = tf.tile(position_embeddings, multiples=[tf.shape(input_ids)[0], 1, 1])
        token_type_embeddings = tf.gather(self.segment_embedding, segment_ids)
        token_embeddings = tf.gather(self.word_embedding, input_ids)

        embeddings = token_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings


class AlbertMultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=8,
                 hidden_dropout_rate=0.0,
                 attention_dropout_rate=0.0,
                 initializer_range=0.02,
                 epsilon=1e-8,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size

        self.query_weight = tf.keras.layers.Dense(self.hidden_size, name='query')
        self.key_weight = tf.keras.layers.Dense(self.hidden_size, name='key')
        self.value_weight = tf.keras.layers.Dense(self.hidden_size, name='value')

        self.dense = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name='dense')
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=epsilon, name='layer_norm')
        self.attention_dropout = tf.keras.layers.Dropout(attention_dropout_rate)
        self.output_dropout = tf.keras.layers.Dropout(hidden_dropout_rate)

    def _split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.hidden_size // self.num_attention_heads))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _scaled_dot_product_attention(self, query, key, value, attention_mask, training=None):
        query = tf.cast(query, dtype=self.dtype)
        key = tf.cast(key, dtype=self.dtype)
        value = tf.cast(value, dtype=self.dtype)

        score = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(query)[-1], self.dtype)
        score = score / tf.math.sqrt(dk)
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, dtype=self.dtype)
            score += tf.cast((1.0 - attention_mask) * -10000.0, self.dtype)
        attn_weights = tf.nn.softmax(score, axis=-1)
        attn_weights = self.attention_dropout(attn_weights, training=training)
        context = tf.matmul(attn_weights, value)
        return context, attn_weights

    def call(self, query, key, value, attention_mask, training=None):
        origin_input = query  # query == key == value

        batch_size = tf.shape(query)[0]
        query = self._split_heads(self.query_weight(query), batch_size)
        key = self._split_heads(self.key_weight(key), batch_size)
        value = self._split_heads(self.value_weight(value), batch_size)

        context, attn_weights = self._scaled_dot_product_attention(query, key, value, attention_mask, training=training)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, [batch_size, -1, self.hidden_size])
        output = self.dense(context)
        output = self.output_dropout(output, training=training)
        output = self.layer_norm(output + origin_input)
        return output, attn_weights


class AlbertEncoderLayer(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=8,
                 intermediate_size=3072,
                 activation='gelu',
                 hidden_dropout_rate=0.0,
                 attention_dropout_rate=0.0,
                 epsilon=1e-8,
                 initializer_range=0.02,
                 **kwargs):
        super().__init__(**kwargs)

        self.attention = AlbertMultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            hidden_dropout_rate=hidden_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            initializer_range=initializer_range,
            epsilon=epsilon,
            name='attention')

        self.ffn = tf.keras.layers.Dense(
            intermediate_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name='ffn')
        self.activation = choose_activation(activation)
        self.ffn_output = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name='ffn_output')
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=epsilon, name='layer_norm')
        self.dropout = tf.keras.layers.Dropout(hidden_dropout_rate)

    def call(self, hidden_states, attention_mask, training=None):
        attn_output, attn_weights = self.attention(
            hidden_states, hidden_states, hidden_states, attention_mask, training=training)
        outputs = self.ffn(attn_output)
        outputs = self.activation(outputs)
        outputs = self.ffn_output(outputs)
        outputs = self.dropout(outputs)
        outputs = self.layer_norm(outputs + attn_output)
        return outputs, attn_weights


class AlbertEncoderGroup(tf.keras.layers.Layer):

    def __init__(self,
                 num_layers_each_group=1,
                 hidden_size=768,
                 num_attention_heads=8,
                 intermediate_size=3072,
                 activation='gelu',
                 hidden_dropout_rate=0.2,
                 attention_dropout_rate=0.1,
                 epsilon=1e-12,
                 initializer_range=0.02,
                 **kwargs):
        super().__init__(**kwargs)

        self.encoder_layers = [
            AlbertEncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                activation=activation,
                hidden_dropout_rate=hidden_dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                epsilon=epsilon,
                initializer_range=initializer_range,
                name='layer_{}'.format(i)
            ) for i in range(num_layers_each_group)
        ]

    def call(self, hidden_states, attention_mask, training=None):
        group_hidden_states, group_attn_weights = [], []
        for idx, encoder in enumerate(self.encoder_layers):
            hidden_states, attn_weights = encoder(hidden_states, attention_mask)
            group_hidden_states.append(hidden_states)
            group_attn_weights.append(attn_weights)

        return hidden_states, group_hidden_states, group_attn_weights


class AlbertEncoder(tf.keras.layers.Layer):

    def __init__(self,
                 num_layers=12,
                 num_groups=1,
                 num_layers_each_group=1,
                 hidden_size=768,
                 num_attention_heads=8,
                 intermediate_size=3072,
                 activation='gelu',
                 hidden_dropout_rate=0.2,
                 attention_dropout_rate=0.1,
                 epsilon=1e-12,
                 initializer_range=0.02,
                 **kwargs):
        super(AlbertEncoder, self).__init__(**kwargs)

        self.embedding_mapping = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name='embedding_mapping'
        )
        self.groups = [
            AlbertEncoderGroup(
                num_layers_each_group=num_layers_each_group,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                activation=activation,
                hidden_dropout_rate=hidden_dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                epsilon=epsilon,
                initializer_range=initializer_range,
                name='group_{}'.format(i),
            ) for i in range(num_groups)
        ]

        self.num_layers = num_layers
        self.num_groups = num_groups

    def call(self, hidden_states, attention_mask, training=None):
        hidden_states = self.embedding_mapping(hidden_states)

        all_hidden_states, all_attention_weights = [], []
        for i in range(self.num_layers):
            layers_per_group = self.num_layers // self.num_groups
            group_index = i // layers_per_group
            hidden_states, group_hidden_states, group_attn_weights = self.groups[group_index](
                hidden_states, attention_mask)
            all_hidden_states.extend(group_hidden_states)
            all_attention_weights.extend(group_attn_weights)

        # stack all_hidden_states to shape:
        # [batch_size, num_layers, num_attention_heads, hidden_size]
        all_hidden_states = tf.stack(all_hidden_states, axis=0)
        all_hidden_states = tf.transpose(all_hidden_states, perm=[1, 0, 2, 3])
        # stack all_attention_scores to shape:
        # [batch_size, num_layers, num_attention_heads, seqeucen_length, sequence_length]
        all_attention_weights = tf.stack(all_attention_weights, axis=0)
        all_attention_weights = tf.transpose(all_attention_weights, perm=[1, 0, 2, 3, 4])
        return hidden_states, all_hidden_states, all_attention_weights


class AlbertPooler(tf.keras.layers.Layer):

    def __init__(self, hidden_size, initializer_range=0.02, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            activation='tanh',
            name='dense')

    def call(self, sequence_output):
        return self.dense(sequence_output[:, 0])


class Albert(tf.keras.Model):

    def __init__(self,
                 vocab_size=-1,
                 max_positions=512,
                 embedding_size=128,
                 type_vocab_size=2,
                 num_layers=12,
                 num_groups=1,
                 num_layers_each_group=1,
                 hidden_size=768,
                 num_attention_heads=8,
                 intermediate_size=3072,
                 activation='gelu',
                 hidden_dropout_rate=0.2,
                 attention_dropout_rate=0.1,
                 epsilon=1e-12,
                 initializer_range=0.02,
                 return_states=False,
                 return_attention_weights=False,
                 **kwargs):
        super(Albert, self).__init__(**kwargs)
        assert vocab_size > 0, "vocab_size must greater than 0."

        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.max_positions = max_positions
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_groups = num_groups
        self.num_layers_each_group = num_layers_each_group
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.initializer_range = initializer_range
        self.initialize_range = initializer_range

        self.embedding = AlbertEmbedding(
            vocab_size=vocab_size,
            max_positions=max_positions,
            embedding_size=embedding_size,
            type_vocab_size=type_vocab_size,
            hidden_dropout_rate=hidden_dropout_rate,
            epsilon=epsilon,
            initializer_range=initializer_range,
            name='embeddings')

        self.encoder = AlbertEncoder(
            num_layers=num_layers,
            num_groups=num_groups,
            num_layers_each_group=num_layers_each_group,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            activation=activation,
            hidden_dropout_rate=hidden_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            epsilon=epsilon,
            initializer_range=initializer_range,
            name='encoder')

        self.pooler = AlbertPooler(hidden_size=hidden_size, initializer_range=initializer_range, name='pooler')

        self.return_states = return_states
        self.return_attention_weights = return_attention_weights

    @tf.function(input_signature=[
        {
            'input_ids': tf.TensorSpec(shape=(None, None), dtype=tf.int32, name='input_ids'),
            'segment_ids': tf.TensorSpec(shape=(None, None), dtype=tf.int32, name='segment_ids'),
            'attention_mask': tf.TensorSpec(shape=(None, None), dtype=tf.int32, name='attention_mask')
        }
    ])
    def serving(self, inputs):
        input_ids, segment_ids, attention_mask = inputs['input_ids'], inputs['segment_ids'], inputs['attention_mask']
        outputs = self(input_ids, segment_ids, attention_mask)
        outputs = list(outputs)
        results = {
            'sequence_output': outputs.pop(0),
            'pooled_output': outputs.pop(0),
        }
        if self.return_states:
            results['hidden_states'] = outputs.pop(0)
        if self.return_attention_weights:
            results['attention_weights'] = outputs.pop(0)
        return results

    def call(self, input_ids, segment_ids=None, attention_mask=None, training=None):
        input_ids, segment_ids, attention_mask = unpack_inputs_3([input_ids, segment_ids, attention_mask])
        attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
        embed = self.embedding(input_ids, segment_ids)
        sequence_outputs, all_hidden_states, all_attn_weights = self.encoder(embed, attention_mask)
        # take [CLS]
        pooled_output = self.pooler(sequence_outputs)
        outputs = (sequence_outputs, pooled_output)
        if self.return_states:
            outputs += (all_hidden_states, )
        if self.return_attention_weights:
            outputs += (all_attn_weights, )
        return outputs

    def dummy_inputs(self):
        input_ids = tf.constant([0] * 128, dtype=tf.int64, shape=(1, 128))
        segment_ids = tf.constant([0] * 128, dtype=tf.int64, shape=(1, 128))
        attn_mask = tf.constant([1] * 128, dtype=tf.int64, shape=(1, 128))
        return input_ids, segment_ids, attn_mask

    @classmethod
    def from_pretrained(cls, pretrained_model_dir, adapter=None, verbose=True, **kwargs):
        config_file, ckpt, vocab_file = parse_pretrained_model_files(pretrained_model_dir)
        if not adapter:
            adapter = AlbertAdapter(**kwargs)
        model_config = adapter.adapte_config(config_file, **kwargs)
        model_config['return_states'] = kwargs.get('return_states', False)
        model_config['return_attention_weights'] = kwargs.get('return_attention_weights', False)
        model = cls(**model_config)
        input_ids, segment_ids, attn_mask = model.dummy_inputs()
        model(input_ids, segment_ids, attn_mask)
        adapter.adapte_weights(model, model_config, ckpt, **kwargs)
        return model

    def get_config(self):
        config = {
            'vocab_size': self.vocab_size,
            'type_vocab_size': self.type_vocab_size,
            'max_positions': self.max_positions,
            'num_layers': self.num_layers,
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'intermediate_size': self.intermediate_size,
            'hidden_dropout_rate': self.hidden_dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'initializer_range': self.initializer_range,
            'return_states': self.return_states,
            'return_attention_weights': self.return_attention_weights
        }
        return config
