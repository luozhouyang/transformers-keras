import json
import logging
import os

import tensorflow as tf

from transformers_keras.adapters import parse_pretrained_model_files
from transformers_keras.adapters.albert_adapter import AlbertAdapter

from .layers import MultiHeadAttention
from .modeling_bert import BertEmbedding, BertEncoderLayer, BertIntermediate
from .modeling_utils import (choose_activation, complete_inputs, initialize,
                             unpack_inputs_2)


class AlbertEmbedding(tf.keras.layers.Layer):

    def __init__(self,
                 vocab_size=-1,
                 max_positions=512,
                 type_vocab_size=2,
                 embedding_size=128,
                 dropout_rate=0.2,
                 stddev=0.02,
                 epsilon=1e-8,
                 **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.initializer_range = stddev

        self.position_embedding = tf.keras.layers.Embedding(
            max_positions,
            embedding_size,
            embeddings_initializer=initialize(stddev),
            name='position_embeddings')
        self.segment_embedding = tf.keras.layers.Embedding(
            type_vocab_size,
            embedding_size,
            embeddings_initializer=initialize(stddev),
            name='token_type_embeddings')

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=epsilon, name='layer_norm')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def build(self, input_shape):
        self.word_embedding = self.add_weight(
            name='weight',
            shape=[self.vocab_size, self.embedding_size],
            initializer=initialize(self.initializer_range))
        super().build(input_shape)

    def call(self, inputs, mode='embedding', training=None):
        if mode == 'linear':
            return tf.matmul(inputs, self.word_embedding, transpose_b=True)

        input_ids, token_type_ids = unpack_inputs_2(inputs)
        seq_len = tf.shape(input_ids)[1]
        position_ids = tf.range(seq_len, dtype=tf.int32)[tf.newaxis, :]

        pos_embedding = self.position_embedding(position_ids)
        seg_embedding = self.segment_embedding(token_type_ids)
        tok_embedding = tf.gather(self.word_embedding, input_ids)

        embedding = tok_embedding + pos_embedding + seg_embedding
        embedding = self.layer_norm(embedding)
        embedding = self.dropout(embedding)
        return embedding


class AlbertMultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=8,
                 hidden_dropout_rate=0.0,
                 attention_dropout_rate=0.0,
                 stddev=0.02,
                 epsilon=1e-8,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size

        self.query_weight = tf.keras.layers.Dense(self.hidden_size, name='query')
        self.key_weight = tf.keras.layers.Dense(self.hidden_size, name='key')
        self.value_weight = tf.keras.layers.Dense(self.hidden_size, name='value')

        self.dense = tf.keras.layers.Dense(hidden_size, kernel_initializer=initialize(stddev), name='dense')
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
        dk = tf.cast(tf.shape(query)[-1], tf.float32)
        score = score / tf.math.sqrt(dk)
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, dtype=self.dtype)
            score += (1.0 - attention_mask) * -10000.0
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
                 stddev=0.02,
                 **kwargs):
        super().__init__(**kwargs)

        self.attention = AlbertMultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            hidden_dropout_rate=hidden_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            stddev=stddev,
            epsilon=epsilon,
            name='attention')

        self.ffn = tf.keras.layers.Dense(intermediate_size, kernel_initializer=initialize(stddev), name='ffn')
        self.activation = choose_activation(activation)
        self.ffn_output = tf.keras.layers.Dense(hidden_size, kernel_initializer=initialize(stddev), name='ffn_output')
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
                 stddev=0.02,
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
                stddev=stddev,
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
                 stddev=0.02,
                 **kwargs):
        super(AlbertEncoder, self).__init__(**kwargs)

        self.embedding_mapping = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=initialize(stddev),
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
                stddev=stddev,
                name='group_{}'.format(i),
            ) for i in range(num_groups)
        ]

        self.num_layers = num_layers
        self.num_groups = num_groups

    def call(self, hidden_states, attention_mask, training=None):
        hidden_states = self.embedding_mapping(hidden_states)

        all_hidden_states, all_attn_weights = [], []
        for i in range(self.num_layers):
            layers_per_group = self.num_layers // self.num_groups
            group_index = i // layers_per_group
            hidden_states, group_hidden_states, group_attn_weights = self.groups[group_index](
                hidden_states, attention_mask)
            all_hidden_states.extend(group_hidden_states)
            all_attn_weights.extend(group_attn_weights)

        return hidden_states, all_hidden_states, all_attn_weights


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
                 stddev=0.02,
                 return_states=False,
                 return_attention_weights=False,
                 **kwargs):
        kwargs.pop('name', None)
        super(Albert, self).__init__(name='albert', **kwargs)
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
        self.stddev = stddev
        self.initialize_range = stddev

        self.embedding = AlbertEmbedding(
            vocab_size=vocab_size,
            max_positions=max_positions,
            embedding_size=embedding_size,
            type_vocab_size=type_vocab_size,
            dropout_rate=hidden_dropout_rate,
            epsilon=epsilon,
            stddev=stddev,
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
            stddev=stddev,
            name='encoder')

        self.pooler = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=initialize(stddev),
            activation='tanh',
            name='pooler'
        )

        self.return_states = return_states
        self.return_attention_weights = return_attention_weights

    def call(self, input_ids, segment_ids, attention_mask, training=None):
        input_ids, segment_ids, attention_mask = complete_inputs(input_ids, segment_ids, attention_mask)

        attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
        embed = self.embedding(inputs=(input_ids, segment_ids), mode='embedding')
        sequence_outputs, all_hidden_states, all_attn_weights = self.encoder(embed, attention_mask)
        # take [CLS]
        pooled_output = self.pooler(sequence_outputs[:, 0])
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
            'stddev': self.stddev,
            'return_states': self.return_states,
            'return_attention_weights': self.return_attention_weights
        }
        return config


class AlbertMLMHead(tf.keras.layers.Layer):

    def __init__(self,
                 embedding,
                 vocab_size=-1,
                 embedding_size=128,
                 activation='gelu',
                 epsilon=1e-12,
                 stddev=0.02,
                 **kwargs):
        super(AlbertMLMHead, self).__init__(**kwargs)
        assert vocab_size > 0, "vocab_size must greater than 0."
        self.vocab_size = vocab_size
        self.embedding = embedding  # use embedding matrix to decode
        self.activation = choose_activation(activation)
        self.stddev = stddev
        self.epsilon = epsilon
        self.dense = tf.keras.layers.Dense(
            embedding_size, kernel_initializer=initialize(self.stddev), name='dense')
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon, name='layer_norm')

    def build(self, input_shape):
        self.decoder_bias = self.add_weight(
            shape=(self.vocab_size,),
            initializer='zeros',
            trainable=True,
            name='decoder/bias'
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        pooled_output = inputs
        output = self.layer_norm(self.activation(self.dense(pooled_output)))
        output = self.embedding(output, mode='linear') + self.decoder_bias
        return output


class AlbertSOPHead(tf.keras.layers.Layer):

    def __init__(self, stddev=0.02, **kwargs):
        super(AlbertSOPHead, self).__init__(**kwargs)
        self.classifier = tf.keras.layers.Dense(2, kernel_initializer=initialize(stddev), name='dense')

    def call(self, inputs, training=None):
        return self.classifier(inputs)
