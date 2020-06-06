import numpy as np
import tensorflow as tf

from .layers import DecoderLayer, EncoderLayer


class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, max_positions=512, embedding_size=512, **kwargs):
        super().__init__(**kwargs)
        self.max_positions = max_positions
        self.embedding_size = embedding_size

    def build(self, input_shape):

        def _initializer(shape, dtype=tf.float32):
            pos = np.arange(self.max_positions)[:, tf.newaxis]
            d = np.arange(self.embedding_size)[tf.newaxis, :]
            rads = 1 / np.power(10000, (2 * (d // 2)) / np.float32(self.embedding_size))
            rads = pos * rads

            rads[:, 0::2] = np.sin(rads[:, 0::2])
            rads[:, 1::2] = np.cos(rads[:, 1::2])

            rads = tf.cast(rads, dtype=dtype)
            rads = tf.reshape(rads, shape=shape)
            return rads

        self.position_embedding = self.add_weight(
            name='position_embedding',
            shape=(self.max_positions, self.embedding_size),
            dtype=tf.float32,
            initializer=_initializer,
            trainable=False)

        super().build(input_shape)

    def call(self, inputs, training=None):
        token_ids = inputs
        pos_embedding = self.position_embedding[tf.newaxis, :]
        embedding = pos_embedding[:, :tf.shape(token_ids)[1], :]
        return embedding

    def get_config(self):
        config = {
            'max_positions': self.max_positions,
            'embedding_size': self.embedding_size
        }
        base = super().get_config()
        return dict(list(base.items()) + list(config.items()))


class TransformerEmbedding(tf.keras.layers.Layer):

    def __init__(self, vocab_size, max_positions=512, embedding_size=512, dropout_rate=0.2, **kwargs):
        super(TransformerEmbedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_positions = max_positions
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.token_embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        self.positional_encoding = PositionalEncoding(self.max_positions, self.embedding_size)

    def call(self, inputs, training=None):
        token_ids = inputs
        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.positional_encoding(token_ids)
        embedding = token_embeddings + position_embeddings
        embedding = self.dropout(embedding, training=training)
        return embedding

    def get_config(self):
        conf = {
            'vocab_size': self.source_vocab_size,
            'max_positions': self.max_positions,
            'embedding_size': self.embedding_size,
            'dropout_rate': self.dropout_rate
        }
        p = super(TransformerEmbedding, self).get_config()
        return dict(list(p.items()) + list(conf.items()))


class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(self,
                 vocab_size,
                 max_positions=512,
                 hidden_size=512,
                 num_layers=6,
                 num_attention_heads=8,
                 ffn_size=2048,
                 dropout_rate=0.2,
                 epsilon=1e-6,
                 **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_positions = max_positions
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.ffn_size = ffn_size
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.embedding = TransformerEmbedding(vocab_size, max_positions, hidden_size, dropout_rate)
        self.encoders = [
            EncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                ffn_size=ffn_size,
                dropout_rate=dropout_rate,
                epsilon=epsilon,
                name='EncoderLayer{}'.format(i)
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
        config = {
            'num_layers': self.num_layers,
            'vocab_size': self.vocab_size,
            'max_positions': self.max_positions,
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'ffn_size': self.ffn_size,
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon
        }
        p = super().get_config()
        return dict(list(p.items()) + list(config.items()))


class TransformerDecoder(tf.keras.layers.Layer):

    def __init__(self,
                 vocab_size,
                 max_positions=512,
                 hidden_size=512,
                 num_layers=6,
                 num_attention_heads=8,
                 ffn_size=2048,
                 dropout_rate=0.2,
                 epsilon=1e-6,
                 **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_positions = max_positions
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.ffn_size = ffn_size
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.embedding = TransformerEmbedding(vocab_size, max_positions, hidden_size, dropout_rate)
        self.decoders = [
            DecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                ffn_size=ffn_size,
                dropout_rate=dropout_rate,
                epsilon=epsilon,
                name='DecoderLayer{}'.format(i)
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
            'vocab_size': self.vocab_size,
            'max_positions': self.max_positions,
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'ffn_size': self.ffn_size,
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon
        }
        p = super(TransformerDecoder, self).get_config()
        return dict(list(p.items()) + list(config.items()))


class Transformer(tf.keras.layers.Layer):

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 max_positions=512,
                 hidden_size=512,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 num_attention_heads=8,
                 ffn_size=2048,
                 dropout_rate=0.2,
                 epsilon=1e-6,
                 **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.encoder = TransformerEncoder(
            src_vocab_size, max_positions, hidden_size,
            num_layers=num_encoder_layers, dropout_rate=dropout_rate, epsilon=epsilon)
        self.decoder = TransformerDecoder(
            tgt_vocab_size, max_positions, hidden_size,
            num_layers=num_encoder_layers, dropout_rate=dropout_rate, epsilon=epsilon)
        self.dense = tf.keras.layers.Dense(tgt_vocab_size)

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
