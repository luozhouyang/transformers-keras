import numpy as np
import tensorflow as tf


class TransformerConfig(object):

    def __init__(self, **kwargs):
        self.num_encoder_layers = kwargs.pop('num_encoder_layers', 6)
        self.num_decoder_layers = kwargs.pop('num_decoder_layers', 6)
        self.num_attention_heads = kwargs.pop('num_attention_heads', 8)
        self.hidden_size = kwargs.pop('hidden_size', 512)
        self.ffn_size = kwargs.pop('ffn_size', 2048)
        self.dropout_rate = kwargs.pop('dropout_rate', 0.1)
        self.source_vocab_size = kwargs.pop('source_vocab_size', 100)
        self.target_vocab_size = kwargs.pop('target_vocab_size', 100)
        self.max_positions = kwargs.pop('max_positions', 512)


class TransformerEmbedding(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super(TransformerEmbedding, self).__init__(**kwargs)
        self.source_vocab_size = config.source_vocab_size
        self.max_positions = config.max_positions
        self.hidden_size = config.hidden_size
        self.dropout_rate = config.dropout_rate
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)
        self.token_embedding = tf.keras.layers.Embedding(self.source_vocab_size, self.hidden_size)

    def build(self, input_shape):

        def _initializer(shape, dtype=tf.float32):
            pos = np.arange(self.max_positions)[:, tf.newaxis]
            d = np.arange(self.hidden_size)[tf.newaxis, :]
            rads = 1 / np.power(10000, (2 * (d // 2)) / np.float32(self.hidden_size))
            rads = pos * rads

            rads[:, 0::2] = np.sin(rads[:, 0::2])
            rads[:, 1::2] = np.cos(rads[:, 1::2])

            rads = tf.cast(rads, dtype=dtype)
            rads = tf.reshape(rads, shape=shape)
            return rads

        with tf.name_scope('position_embedding'):
            self.position_embedding = self.add_weight(
                name='position_embedding',
                shape=(self.max_positions, self.hidden_size),
                dtype=tf.float32,
                initializer=_initializer,
                trainable=False
            )

        super(TransformerEmbedding, self).build(input_shape)

    def call(self, inputs, training=None):
        token_ids = inputs
        token_embeddings = self.token_embedding(token_ids)
        pos_embedding = self.position_embedding[tf.newaxis, :]
        position_embeddings = pos_embedding[:, :tf.shape(token_ids)[1], :]
        embedding = token_embeddings + position_embeddings
        embedding = self.dropout(embedding, training=training)
        return embedding

    def get_config(self):
        conf = {
            'source_vocab_size': self.source_vocab_size,
            'max_positions': self.max_positions,
            'hidden_size': self.hidden_size,
            'dropout_rate': self.dropout_rate
        }
        p = super(TransformerEmbedding, self).get_config()
        return dict(list(p.items()) + list(conf.items()))


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        assert self.hidden_size % self.num_attention_heads == 0
        self.depth = self.hidden_size // self.num_attention_heads

        self.wq = tf.keras.layers.Dense(self.hidden_size)
        self.wk = tf.keras.layers.Dense(self.hidden_size)
        self.wv = tf.keras.layers.Dense(self.hidden_size)

        self.dropout_rate = config.dropout_rate
        self.attention_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense = tf.keras.layers.Dense(self.hidden_size)

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12)

    def call(self, inputs, attn_mask=None, training=None):
        query, key, value = inputs
        residual = query
        batch_size = tf.shape(query)[0]

        def _split_heads(x):
            x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        query = _split_heads(self.wq(query))
        key = _split_heads(self.wk(key))
        value = _split_heads(self.wv(value))

        def _scaled_dot_product_attention(q, k, v, mask=None):
            s = tf.matmul(q, k, transpose_b=True)
            dk = tf.cast(tf.shape(q)[-1], dtype=tf.float32)
            s = s/tf.math.sqrt(dk)
            if mask is not None:
                s += mask * -10000.0
            attn = tf.nn.softmax(s)
            attn = self.attention_dropout(attn, training=training)
            context = tf.matmul(attn, v)
            context = tf.transpose(context, perm=[0, 2, 1, 3])
            context = tf.reshape(context, shape=(batch_size, -1, self.hidden_size))
            return context, attn

        context, attn_weights = _scaled_dot_product_attention(query, key, value, attn_mask)
        outputs = self.dense(context)
        outputs = self.dropout(outputs, training=training)
        outputs = self.layer_norm(residual + outputs)
        return outputs, attn_weights

    def get_config(self):
        config = {
            'num_attention_heads': self.num_attention_heads,
            'hidden_size': self.hidden_size,
            'depth': self.depth,
            'dropout_rate': self.dropout_rate,
        }
        p = super(MultiHeadAttention, self).get_config()
        return dict(list(p.items()) + list(config.items()))


class PairWiseFeedForwardNetwork(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super(PairWiseFeedForwardNetwork, self).__init__(**kwargs)
        self.ffn_size = config.ffn_size
        self.hidden_size = config.hidden_size
        self.dropout_rate = config.dropout_rate
        self.dense1 = tf.keras.layers.Dense(self.ffn_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.hidden_size)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12)

    def call(self, inputs, training=None):
        outputs = self.dense2(self.dense1(inputs))
        outputs = self.dropout(outputs, training=training)
        outputs = self.layer_norm(outputs + inputs)
        return outputs

    def get_config(self):
        config = {
            'ffn_size': self.ffn_size,
            'hidden_size': self.hidden_size,
        }
        p = super(PairWiseFeedForwardNetwork, self).get_config()
        return dict(list(p.items()) + list(config.items()))


class TransformerEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.ffn_size = config.ffn_size
        self.dropout_rate = config.dropout_rate

        self.mha = MultiHeadAttention(config)
        self.ffn = PairWiseFeedForwardNetwork(config)

    def call(self, inputs, attn_mask=None):
        hidden_states, attn_weights = self.mha(inputs=inputs, attn_mask=attn_mask)
        outputs = self.ffn(hidden_states)
        return outputs, attn_weights

    def get_config(self):
        config = {
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'ffn_size': self.ffn_size,
            'dropout_rate': self.dropout_rate
        }
        p = super(TransformerEncoderLayer, self).get_config()
        return dict(list(p.items()) + list(config.items()))


class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_layers = config.num_encoder_layers
        self.embedding = TransformerEmbedding(config)
        self.encoders = [
            TransformerEncoderLayer(config) for _ in range(self.num_layers)
        ]

    def call(self, inputs):
        token_ids = inputs
        embeddings = self.embedding(inputs=token_ids)

        def _create_mask(x):
            mask = tf.cast(tf.equal(0, x), dtype=tf.float32)
            mask = mask[:, tf.newaxis, tf.newaxis, :]
            return mask

        attn_weights = []
        outputs = embeddings
        attn_mask = _create_mask(token_ids)
        for i in range(self.num_layers):
            outputs, weights = self.encoders[i](inputs=(outputs, outputs, outputs), attn_mask=attn_mask)
            attn_weights.append(weights)

        return outputs, attn_weights

    def get_config(self):
        conf = {
            'num_layers': self.num_layers,
        }
        p = super().get_config()
        return dict(list(p.items()) + list(conf.items()))


class TransformerDecoderLayer(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super(TransformerDecoderLayer, self).__init__(name='TransformerDecoderLayer', **kwargs)
        self.mha1 = MultiHeadAttention(config)
        self.mha2 = MultiHeadAttention(config)
        self.ffn = PairWiseFeedForwardNetwork(config)
        self.dropout_rate = config.dropout_rate
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12)

    def call(self, inputs, look_ahead_mask=None, padding_mask=None, training=None):
        x, enc_output = inputs
        dec_outputs_1, self_attention_weights = self.mha1(inputs=(x, x, x), attn_mask=look_ahead_mask)
        dec_outputs_2, context_attention_weights = self.mha2(
            inputs=(enc_output, enc_output, dec_outputs_1), attn_mask=padding_mask)

        outputs = self.ffn(dec_outputs_2)
        outputs = self.dropout(outputs, training=training)
        outputs = self.layer_norm(outputs + dec_outputs_2)
        return outputs, self_attention_weights, context_attention_weights

    def get_config(self):
        config = {
            'dropout_rate': self.dropout_rate,
        }
        p = super(TransformerDecoderLayer, self).get_config()
        return dict(list(p.items()) + list(config.items()))


class TransformerDecoder(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super(TransformerDecoder, self).__init__(name='TransformerDecoder', **kwargs)
        self.num_layers = config.num_decoder_layers
        self.hidden_size = config.hidden_size
        self.embedding = TransformerEmbedding(config)
        self.decoders = [
            TransformerDecoderLayer(config) for _ in range(self.num_layers)
        ]

    def call(self, inputs):
        token_ids, enc_outputs = inputs
        embeddings = self.embedding(token_ids)

        def _create_mask(x):
            padding_mask = tf.cast(tf.equal(0, x), dtype=tf.float32)
            padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
            size = tf.shape(x)[1]
            look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
            look_ahead_mask = tf.maximum(look_ahead_mask, padding_mask)
            return look_ahead_mask, padding_mask

        self_attn_weights, context_attn_weights = [], []
        outputs = embeddings
        look_ahead_mask, padding_mask = _create_mask(token_ids)
        for i in range(self.num_layers):
            outputs, self_attn, context_attn = self.decoders[i](
                inputs=(outputs, enc_outputs), look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
            self_attn_weights.append(self_attn)
            context_attn_weights.append(context_attn)

        return outputs, self_attn_weights, context_attn_weights

    def get_config(self):
        config = {
            'num_layers': self.num_layers,
        }
        p = super(TransformerDecoder, self).get_config()
        return dict(list(p.items()) + list(config.items()))


class Transformer(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.dense = tf.keras.layers.Dense(config.target_vocab_size)

    def call(self, inputs):
        x_ids, y_ids = inputs
        enc_outputs, enc_attns = self.encoder(x_ids)
        dec_outputs, dec_attns_0, dec_attns_1 = self.decoder(inputs=(y_ids, enc_outputs))
        logits = self.dense(dec_outputs)
        return logits, enc_attns[-1], dec_attns_0[-1], dec_attns_1[-1]
