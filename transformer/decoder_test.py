import tensorflow as tf

from transformer.decoder import Decoder, DecoderLayer
from transformer.encoder import Encoder, EncoderLayer


class DecoderTest(tf.test.TestCase):

    def testDecoderLayer(self):
        sample_encoder_layer = EncoderLayer(512, 8, 2048)
        sample_encoder_layer_output = sample_encoder_layer(
            tf.random.uniform((64, 43, 512)), False, None)

        print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)

        sample_decoder_layer = DecoderLayer(512, 8, 2048)
        sample_decoder_layer_output, _, _ = sample_decoder_layer(
            tf.random.uniform((64, 50, 512)), sample_encoder_layer_output, False, None, None)

        print(sample_decoder_layer_output.shape)  # (batch_size, target_seq_len, d_model)

    def testDecoder(self):
        sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=8500)
        sample_encoder_output = sample_encoder(tf.random.uniform((64, 62)), training=False, mask=None)

        print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)

        sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8, dff=2048, target_vocab_size=8000)
        output, attn = sample_decoder(
            tf.random.uniform((64, 26)),
            enc_output=sample_encoder_output,
            training=False, look_ahead_mask=None,
            padding_mask=None)

        print(output.shape)
        print(attn['decoder_layer2_block2'].shape)


if __name__ == '__main__':
    tf.test.main()
