import json
import logging
import os

import tensorflow as tf

from transformers_keras.adapters import parse_pretrained_model_files
from transformers_keras.adapters.bert_adapter import BertAdapter

from .modeling_utils import choose_activation, unpack_inputs_3


class BertEmbedding(tf.keras.layers.Layer):

    def __init__(self,
                 vocab_size=21128,
                 max_positions=512,
                 embedding_size=768,
                 type_vocab_size=2,
                 hidden_dropout_rate=0.1,
                 initializer_range=0.02,
                 epsilon=1e-12,
                 **kwargs):
        super().__init__(**kwargs)
        assert vocab_size > 0, "vocab_size must greater than 0."
        self.vocab_size = vocab_size
        self.max_positions = max_positions
        self.type_vocab_size = type_vocab_size
        self.embedding_size = embedding_size
        self.initializer_range = initializer_range
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=epsilon, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(hidden_dropout_rate)

    def build(self, input_shape):
        self.token_embedding = self.add_weight(
            'word_embeddings',
            shape=[self.vocab_size, self.embedding_size],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range),
        )
        self.position_embedding = self.add_weight(
            'position_embeddings',
            shape=[self.max_positions, self.embedding_size],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range),
        )
        self.token_type_embedding = self.add_weight(
            'token_type_embeddings',
            shape=[self.type_vocab_size, self.embedding_size],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range),
        )
        return super().build(input_shape)

    @property
    def embedding_table(self):
        return self.token_embedding

    def call(self, input_ids, segment_ids=None, position_ids=None, training=None):
        if segment_ids is None:
            segment_ids = tf.zeros_like(input_ids)
        if position_ids is None:
            position_ids = tf.range(0, tf.shape(input_ids)[1], dtype=input_ids.dtype)
            position_ids = tf.expand_dims(position_ids, axis=0)

        position_embeddings = tf.gather(self.position_embedding, position_ids)
        position_embeddings = tf.tile(position_embeddings, multiples=[tf.shape(input_ids)[0], 1, 1])
        token_type_embeddings = tf.gather(self.token_type_embedding, segment_ids)
        token_embeddings = tf.gather(self.token_embedding, input_ids)

        embeddings = token_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings


class BertMultiHeadAtttetion(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=8,
                 attention_dropout_rate=0.1,
                 initializer_range=0.02,
                 epsilon=1e-8,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.query_weight = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name='query')
        self.key_weight = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name='key')
        self.value_weight = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name='value')
        self.attention_dropout = tf.keras.layers.Dropout(attention_dropout_rate)

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
            score += tf.cast((1.0 - attention_mask) * -10000.0, dtype=self.dtype)
        attn_weights = tf.nn.softmax(score, axis=-1)
        attn_weights = self.attention_dropout(attn_weights, training=training)
        context = tf.matmul(attn_weights, value)
        return context, attn_weights

    def call(self, query, key, value, attention_mask, training=None):
        batch_size = tf.shape(query)[0]
        query = self._split_heads(self.query_weight(query), batch_size)
        key = self._split_heads(self.key_weight(key), batch_size)
        value = self._split_heads(self.value_weight(value), batch_size)
        context, attn_weights = self._scaled_dot_product_attention(
            query, key, value, attention_mask, training=training)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, [batch_size, -1, self.hidden_size])
        return context, attn_weights


class BertAttentionOutput(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_size=768,
                 hidden_dropout_rate=0.1,
                 initializer_range=0.02,
                 epsilon=1e-5,
                 **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name='dense')
        self.dropout = tf.keras.layers.Dropout(hidden_dropout_rate)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=epsilon, name='LayerNorm')

    def call(self, input_states, hidden_states, training=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.layernorm(hidden_states + input_states)
        return hidden_states


class BertAttention(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=8,
                 hidden_dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 initializer_range=0.02,
                 epsilon=1e-5,
                 **kwargs):
        super().__init__(**kwargs)
        self.attention = BertMultiHeadAtttetion(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout_rate=attention_dropout_rate,
            initializer_range=initializer_range,
            epsilon=epsilon,
            name='self')
        self.attention_output = BertAttentionOutput(
            hidden_size=hidden_size,
            hidden_dropout_rate=hidden_dropout_rate,
            initializer_range=initializer_range,
            epsilon=epsilon,
            name='output')

    def call(self, hidden_states, attention_mask, training=None):
        context, attention_weights = self.attention(
            hidden_states, hidden_states, hidden_states, attention_mask, training=training)
        outputs = self.attention_output(hidden_states, context, training=training)
        return outputs, attention_weights


class BertIntermediate(tf.keras.layers.Layer):

    def __init__(self, intermediate_size=3072, activation='gelu', initializer_range=0.02, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            intermediate_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name='dense')
        self.activation = choose_activation(activation)

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class BertIntermediateOutput(tf.keras.layers.Layer):

    def __init__(self, hidden_size=768, hidden_dropout_rate=0.1, initializer_range=0.02, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name='dense')
        self.dropout = tf.keras.layers.Dropout(hidden_dropout_rate)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=epsilon, name='LayerNorm')

    def call(self, input_states, hidden_states, training=None):
        hidden_states = self.dropout(self.dense(hidden_states), training=training)
        hidden_states = self.layernorm(hidden_states + input_states)
        return hidden_states


class BertEncoderLayer(tf.keras.layers.Layer):

    def __init__(self,
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
        # attention block
        self.attention = BertAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            hidden_dropout_rate=hidden_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            initializer_range=initializer_range,
            name='attention')
        # intermediate block
        self.intermediate = BertIntermediate(
            intermediate_size=intermediate_size,
            activation=activation,
            initializer_range=initializer_range,
            name='intermediate')
        # output block
        self.intermediate_output = BertIntermediateOutput(
            hidden_size=hidden_size,
            hidden_dropout_rate=hidden_dropout_rate,
            initializer_range=initializer_range,
            epsilon=epsilon,
            name='output')

    def call(self, hidden_states, attn_mask, training=None):
        attn_output, attn_weights = self.attention(hidden_states, attn_mask, training=training)
        outputs = self.intermediate(attn_output)
        outputs = self.intermediate_output(attn_output, outputs, training=training)
        return outputs, attn_weights


class BertEncoder(tf.keras.layers.Layer):

    def __init__(self,
                 num_layers=6,
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
            BertEncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                activation=activation,
                hidden_dropout_rate=hidden_dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                epsilon=epsilon,
                initializer_range=initializer_range,
                name='layer_{}'.format(i)
            ) for i in range(num_layers)
        ]

    def call(self, hidden_states, attention_mask, training=None):
        all_hidden_states = []
        all_attention_scores = []
        for _, encoder in enumerate(self.encoder_layers):
            hidden_states, attention_score = encoder(hidden_states, attention_mask, training=training)
            all_hidden_states.append(hidden_states)
            all_attention_scores.append(attention_score)
        # stack all_hidden_states to shape:
        # [batch_size, num_layers, num_attention_heads, hidden_size]
        all_hidden_states = tf.stack(all_hidden_states, axis=0)
        all_hidden_states = tf.transpose(all_hidden_states, perm=[1, 0, 2, 3])
        # stack all_attention_scores to shape:
        # [batch_size, num_layers, num_attention_heads, seqeucen_length, sequence_length]
        all_attention_scores = tf.stack(all_attention_scores, axis=0)
        all_attention_scores = tf.transpose(all_attention_scores, perm=[1, 0, 2, 3, 4])
        return hidden_states, all_hidden_states, all_attention_scores


class BertPooler(tf.keras.layers.Layer):

    def __init__(self, hidden_size=768, initializer_range=0.02, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            activation='tanh',
            name='dense')

    def call(self, inputs):
        hidden_states = inputs
        # pool the first token: [CLS]
        outputs = self.dense(hidden_states[:, 0])
        return outputs


class BertPretrainedModel(tf.keras.Model):

    def __init__(self, return_states=False, return_attention_weights=False, **kwargs):
        unused_keys = []
        for k, v in kwargs.items():
            if str(k).startswith('skip_'):
                unused_keys.append(k)
        for k in unused_keys:
            kwargs.pop(k, None)
        super().__init__(**kwargs)
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
        return self.forward(inputs)

    def forward(self, inputs):
        """Forward pass for serving.

        Arguments:
            inputs: A dict, contains `input_ids`, `segment_ids` and `attention_mask`

        Returns:
            outputs: A dict, contains outputs from serving
        """
        raise NotImplementedError()

    def dummy_inputs(self):
        input_ids = tf.constant([0] * 128, dtype=tf.int32, shape=(1, 128))
        segment_ids = tf.constant([0] * 128, dtype=tf.int32, shape=(1, 128))
        attn_mask = tf.constant([1] * 128, dtype=tf.int32, shape=(1, 128))
        return input_ids, segment_ids, attn_mask

    @classmethod
    def _build_extra_config(cls, model_config, **kwargs):
        extra_config = {}
        for k, v in kwargs.items():
            if k in ['verbose']:
                continue
            if k not in model_config:
                if str(k).startswith('override_'):
                    k = k.lstrip('override_')
                extra_config[k] = v
        if extra_config:
            logging.info('Load extra config: \n%s', json.dumps(extra_config, indent=4))
        return extra_config

    @classmethod
    def _merge_config(cls, model_config, extra_config):
        mixed_config = {}
        mixed_config.update(**model_config)
        mixed_config.update(**extra_config)
        logging.info('Using mixed config: \n%s', json.dumps(mixed_config, indent=4))
        return mixed_config

    @classmethod
    def from_pretrained(cls, pretrained_model_dir, adapter=None, **kwargs):
        config_file, ckpt, vocab_file = parse_pretrained_model_files(pretrained_model_dir)
        if not adapter:
            adapter = BertAdapter(**kwargs)
        model_config = adapter.adapte_config(config_file, **kwargs)
        logging.info('Load model config: \n%s', json.dumps(model_config, indent=4))
        extra_config = cls._build_extra_config(model_config, **kwargs)
        mixed_config = cls._merge_config(model_config, extra_config)
        model = cls(**mixed_config)
        input_ids, segment_ids, attn_mask = model.dummy_inputs()
        model(inputs=[input_ids, segment_ids, attn_mask], training=False)
        adapter.adapte_weights(
            model.bert_model,
            model_config,
            ckpt,
            model_prefix=model.name_prefix,
            **kwargs)
        return model

    @classmethod
    def from_config_file(cls, config_file, **kwargs):
        with open(config_file, mode='rt', encoding='utf-8') as fin:
            model_config = json.load(fin)
        logging.info('Load model config: \n%s', json.dumps(model_config, indent=4))
        extra_config = cls._build_extra_config(model_config, **kwargs)
        mixed_config = cls._merge_config(model_config, extra_config)
        model = cls(**mixed_config)
        input_ids, segment_ids, attn_mask = model.dummy_inputs()
        model(input_ids, segment_ids, attn_mask, training=False)
        return model

    @property
    def bert_model(self):
        return self

    @property
    def name_prefix(self):
        return ''


class Bert(BertPretrainedModel):

    def __init__(self,
                 vocab_size=21128,
                 max_positions=512,
                 hidden_size=768,
                 type_vocab_size=2,
                 num_layers=6,
                 num_attention_heads=8,
                 intermediate_size=3072,
                 activation='gelu',
                 hidden_dropout_rate=0.2,
                 attention_dropout_rate=0.1,
                 initializer_range=0.02,
                 epsilon=1e-12,
                 return_states=False,
                 return_attention_weights=False,
                 **kwargs):
        super().__init__(
            return_states=return_states,
            return_attention_weights=return_attention_weights,
            **kwargs)

        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.max_positions = max_positions
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.initializer_range = initializer_range
        self.initialize_range = initializer_range

        self.bert_embedding = BertEmbedding(
            vocab_size=vocab_size,
            max_positions=max_positions,
            embedding_size=hidden_size,
            type_vocab_size=type_vocab_size,
            hidden_dropout_rate=hidden_dropout_rate,
            initializer_range=initializer_range,
            epsilon=epsilon,
            name='embeddings')

        self.bert_encoder = BertEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            activation=activation,
            hidden_dropout_rate=hidden_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            epsilon=epsilon,
            initializer_range=initializer_range,
            name='encoder')

        self.bert_pooler = BertPooler(hidden_size=hidden_size, initializer_range=initializer_range, name='pooler')

    def get_embedding_table(self):
        return self.bert_embedding.embedding_table

    def forward(self, inputs):
        input_ids, segment_ids, attention_mask = inputs['input_ids'], inputs['segment_ids'], inputs['attention_mask']
        outputs = self(inputs=[input_ids, segment_ids, attention_mask])
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

    def call(self, inputs, training=None):
        input_ids, segment_ids, attention_mask = unpack_inputs_3(inputs)
        embedding = self.bert_embedding(input_ids, segment_ids, training=training)
        # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
        attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
        output, all_hidden_states, all_attention_scores = self.bert_encoder(
            embedding, attention_mask, training=training)
        pooled_output = self.bert_pooler(output)
        outputs = (output, pooled_output)
        if self.return_states:
            outputs += (all_hidden_states, )
        if self.return_attention_weights:
            outputs += (all_attention_scores, )
        return outputs

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
