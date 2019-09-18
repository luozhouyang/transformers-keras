import tensorflow as tf

from transformer.transformer_model import Transformer


class TransformerModelTest(tf.test.TestCase):

    def testTransformerModel(self):
        sample_transformer = Transformer(
            num_layers=2, d_model=512, num_heads=8, dff=2048,
            input_vocab_size=8500, target_vocab_size=8000)

        temp_input = tf.random.uniform((64, 62))
        temp_target = tf.random.uniform((64, 26))

        fn_out, _ = sample_transformer(
            inputs=(temp_input, temp_target),
            training=False,
            mask=(None, None, None))

        print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)

        sample_transformer.fit(temp_input, temp_target, batch_size=1)


if __name__ == '__main__':
    tf.test.main()
