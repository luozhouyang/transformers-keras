import numpy as np

from .layers import *


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
        self.epsilon = kwargs.get('epsilon', 1e-6)


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


class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_layers = config.num_encoder_layers
        self.embedding = TransformerEmbedding(config)
        self.encoders = [
            EncoderLayer(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                ffn_size=config.ffn_size,
                dropout_rate=config.dropout_rate,
                epsilon=config.epsilon,
                name='encoder_layer_{}'.format(i)
            ) for i in range(self.num_layers)
        ]

    def call(self, inputs, training=None):
        token_ids, mask = inputs
        embeddings = self.embedding(inputs=token_ids)

        attn_weights = []
        outputs = embeddings
        for i in range(self.num_layers):
            encoder = self.encoders[i]
            outputs, weights = encoder(inputs=(outputs, outputs, outputs, mask))
            attn_weights.append(weights)

        return outputs, attn_weights

    def get_config(self):
        conf = {
            'num_layers': self.num_layers,
        }
        p = super().get_config()
        return dict(list(p.items()) + list(conf.items()))


class TransformerDecoder(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super(TransformerDecoder, self).__init__(name='TransformerDecoder', **kwargs)
        self.num_layers = config.num_decoder_layers
        self.hidden_size = config.hidden_size
        self.embedding = TransformerEmbedding(config)
        self.decoders = [
            DecoderLayer(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                ffn_size=config.ffn_size,
                dropout_rate=config.dropout_rate,
                epsilon=config.epsilon,
                name='decoder_layer_{}'.format(i)
            ) for i in range(self.num_layers)
        ]

    def call(self, inputs, training=None):
        token_ids, enc_outputs, look_ahead_mask, padding_mask = inputs
        embeddings = self.embedding(token_ids)

        self_attn_weights, context_attn_weights = [], []
        outputs = embeddings

        for i in range(self.num_layers):
            decoder = self.decoders[i]
            outputs, self_attn, context_attn = decoder(inputs=(outputs, enc_outputs, look_ahead_mask, padding_mask))
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

    def call(self, inputs, training=None):
        x_ids, y_ids = inputs

        def _create_padding_mask(x):
            mask = tf.cast(tf.equal(0, x), dtype=tf.float32)
            mask = mask[:, tf.newaxis, tf.newaxis, :]
            return mask

        def _create_look_ahead_mask(size):
            mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
            return mask

        def _create_masks(x, y):
            _enc_padding_mask = _create_padding_mask(x)
            _dec_padding_mask = _create_padding_mask(x)
            _look_ahead_mask = _create_look_ahead_mask(tf.shape(y)[1])
            _target_padding_mask = _create_padding_mask(y)
            combined = tf.maximum(_look_ahead_mask, _target_padding_mask)
            return _enc_padding_mask, combined, _dec_padding_mask

        enc_padding_mask, dec_look_ahead_mask, dec_padding_mask = _create_masks(x_ids, y_ids)

        enc_outputs, enc_attns = self.encoder(inputs=(x_ids, enc_padding_mask))

        dec_outputs, dec_attns_0, dec_attns_1 = self.decoder(
            inputs=(y_ids, enc_outputs, dec_look_ahead_mask, dec_padding_mask))

        logits = self.dense(dec_outputs)
        return logits, enc_attns, dec_attns_0, dec_attns_1

    def get_config(self):
        return super().get_config()
