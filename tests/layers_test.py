import tensorflow as tf
from transformers_keras.layers import (DecoderLayer, EncoderLayer,
                                       MultiHeadAttention,
                                       ScaledDotProductAttention)


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


class LayersTest(tf.test.TestCase):

    def testScaledDotProductAttention(self):
        attention = ScaledDotProductAttention()
        x = tf.random.uniform(shape=(4, 16, 64))
        mask = tf.cast(tf.linalg.band_part(
            tf.ones(shape=(4, 16, 16)), num_lower=0, num_upper=-1), dtype=tf.float32)
        context, attn_weight = attention(x, x, x, mask, training=False)
        self.assertEqual([4, 16, 64], context.shape)
        self.assertEqual([4, 16, 16], attn_weight.shape)

    def testMultiHeadAttention(self):
        attention = MultiHeadAttention(hidden_size=512, num_attention_heads=8)
        x = tf.random.uniform(shape=(4, 16, 512))
        mask = tf.cast(tf.linalg.band_part(
            tf.ones(shape=(4, 1, 16, 16)), num_lower=0, num_upper=-1), dtype=tf.float32)
        context, attn_weight = attention(x, x, x, mask, training=False)
        self.assertEqual([4, 16, 512], context.shape)
        self.assertEqual([4, 8, 16, 16], attn_weight.shape)

    def testEncoderLayer(self):
        encoder_layer = EncoderLayer(hidden_size=512, num_attention_heads=8, ffn_size=2048)
        x = tf.random.uniform(shape=(2, 5, 512))
        padding_mask = tf.constant([[0, 0, 0, 1, 1], [0, 0, 0, 0, 1]], shape=(2, 5), dtype=tf.float32)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]

        outputs, attn_weights = encoder_layer(x, x, x, padding_mask)
        self.assertEqual([2, 5, 512], outputs.shape)
        self.assertEqual([2, 8, 5, 5], attn_weights.shape)

    def testDecoderLayer(self):
        decoder_layer = DecoderLayer(hidden_size=512, num_attention_heads=8, ffn_size=2048)
        x = tf.constant([[1, 2, 3, 4, 0], [2, 3, 3, 0, 0]], shape=(2, 5), dtype=tf.int64)
        y = tf.constant([[1, 2, 3, 4, 1, 0, 0], [2, 3, 3, 4, 5, 6, 7]], shape=(2, 7), dtype=tf.int64)

        _, look_ahead_mask, padding_mask = _create_masks(x, y)

        enc_outputs = tf.random.uniform(shape=(2, 5, 512))

        dec_inputs = tf.random.uniform((2, 7, 512))
        dec_outputs, attn_weights1, attn_weights2 = decoder_layer(
            dec_inputs, enc_outputs, look_ahead_mask, padding_mask)

        self.assertEqual([2, 7, 512], dec_outputs.shape)
        self.assertEqual([2, 8, 7, 7], attn_weights1.shape)
        self.assertEqual([2, 8, 7, 5], attn_weights2.shape)


if __name__ == "__main__":
    tf.test.main()
