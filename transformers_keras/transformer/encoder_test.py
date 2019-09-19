import tensorflow as tf

from transformers_keras.transformer.encoder import EncoderLayer, Encoder


class EncoderTest(tf.test.TestCase):

    def testEncoderLayer(self):
        sample_encoder_layer = EncoderLayer(512, 8, 2048)

        output, attn_weights = sample_encoder_layer(
            tf.random.uniform((64, 43, 512)), False, None)

        self.assertAllEqual(output.shape, [64, 43, 512])  # (batch_size, input_seq_len, d_model)
        self.assertAllEqual(attn_weights.shape, [64, 8, 43, 43])  # (batch_size, num_heads, seq_length, seq_length)

    def testEncoder(self):
        sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=8500)
        output, attn_weights = sample_encoder(tf.random.uniform((64, 62)), training=False, mask=None)

        self.assertAllEqual(output.shape, [64, 62, 512])  # (batch_size, input_seq_len, d_model)

        # each layers attention weights
        for k, v in attn_weights.items():
            self.assertAllEqual(v.shape, [64, 8, 62, 62])
            print(k, v.shape)


if __name__ == '__main__':
    tf.test.main()
