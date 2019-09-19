import tensorflow as tf
import numpy as np

from transformers_keras.transformer.transformer_model import Transformer


class TransformerModelTest(tf.test.TestCase):

    def testTransformerModel(self):
        sample_transformer = Transformer(
            num_layers=2, d_model=512, num_heads=8, dff=2048,
            input_vocab_size=503, target_vocab_size=503)

        temp_input = np.random.randint(100, size=(64, 62))
        sos = np.full((64, 1), 501)
        temp_input = np.concatenate((sos, temp_input), axis=1)
        print(temp_input)
        temp_target = np.random.randint(100, size=(64, 26))
        eos = np.full((64, 1), 502)
        temp_target = np.concatenate((temp_target, eos), axis=1)
        print(temp_target)

        fn_out, enc_attn_weights, dec_attn_weights = sample_transformer(
            inputs=(temp_input, temp_target),
            training=False,
            mask=(None, None, None))

        self.assertAllEqual(fn_out.shape, [64, 27, 503])  # (batch_size, tar_seq_len, target_vocab_size)

        sample_transformer.summary()


if __name__ == '__main__':
    tf.test.main()
