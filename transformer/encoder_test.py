import tensorflow as tf

from transformer.encoder import EncoderLayer, Encoder


class EncoderTest(tf.test.TestCase):

    def testEncoderLayer(self):
        sample_encoder_layer = EncoderLayer(512, 8, 2048)

        sample_encoder_layer_output = sample_encoder_layer(
            tf.random.uniform((64, 43, 512)), False, None)

        print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)

    def testEncoder(self):
        sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,dff=2048, input_vocab_size=8500)
        sample_encoder_output = sample_encoder(tf.random.uniform((64, 62)),training=False, mask=None)

        print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)


if __name__ == '__main__':
    tf.test.main()
