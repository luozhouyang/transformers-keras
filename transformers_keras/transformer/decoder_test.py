import tensorflow as tf

from transformers_keras.transformer.decoder import Decoder, DecoderLayer
from transformers_keras.transformer.encoder import Encoder, EncoderLayer


class DecoderTest(tf.test.TestCase):

    def testDecoderLayer(self):
        sample_encoder_layer = EncoderLayer(512, 8, 2048)
        encoder_layer_output, encoder_attn_weights = sample_encoder_layer(
            tf.random.uniform((64, 43, 512)), False, None)

        self.assertAllEqual(encoder_layer_output.shape, [64, 43, 512])  # (batch_size, input_seq_len, d_model)
        self.assertAllEqual(encoder_attn_weights.shape, [64, 8, 43, 43])

        sample_decoder_layer = DecoderLayer(512, 8, 2048)
        decoder_layer_output, decoder_attn_block_1, decoder_attn_block_2 = sample_decoder_layer(
            inputs=(tf.random.uniform((64, 50, 512)), encoder_layer_output),
            training=True,
            mask=(None, None))

        self.assertAllEqual(decoder_layer_output.shape, [64, 50, 512])  # (batch_size, target_seq_len, d_model)
        self.assertAllEqual(decoder_attn_block_1.shape, [64, 8, 50, 50])
        self.assertAllEqual(decoder_attn_block_2.shape, [64, 8, 50, 43])

    def testDecoder(self):
        sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=8500)
        sample_encoder_output, encoder_attn_weights = sample_encoder(
            tf.random.uniform((64, 62)), training=False, mask=None)

        self.assertAllEqual(sample_encoder_output.shape, [64, 62, 512])  # (batch_size, input_seq_len, d_model)
        for k, v in encoder_attn_weights.items():
            self.assertAllEqual(v.shape, [64, 8, 62, 62])
            print(k, v.shape)

        sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8, dff=2048, target_vocab_size=8000)
        sample_decoder_output, decoder_attn_weights = sample_decoder(
            inputs=(tf.random.uniform((64, 26)), sample_encoder_output),
            training=True,
            mask=(None, None))

        self.assertAllEqual(sample_decoder_output.shape, [64, 26, 512])
        # num_layers=2
        for i in range(2):
            k1 = 'decoder_layer%d_block1' % (i + 1)
            self.assertAllEqual(decoder_attn_weights[k1].shape, [64, 8, 26, 26])
            k2 = 'decoder_layer%d_block2' % (i + 1)
            self.assertAllEqual(decoder_attn_weights[k2].shape, [64, 8, 26, 62])
        print('decoder attention weights:')
        for k, v in decoder_attn_weights.items():
            print(k, v.shape)


if __name__ == '__main__':
    tf.test.main()
