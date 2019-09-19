import numpy as np
import tensorflow as tf

from transformers_keras.transformer import funcs


class FuncsTest(tf.test.TestCase):

    def testScaledDotProductAttention(self):
        np.set_printoptions(suppress=True)

        temp_k = tf.constant([[10, 0, 0],
                              [0, 10, 0],
                              [0, 0, 10],
                              [0, 0, 10]], dtype=tf.float32)  # (4, 3)

        temp_v = tf.constant([[1, 0],
                              [10, 0],
                              [100, 5],
                              [1000, 6]], dtype=tf.float32)  # (4, 3)
        temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)

        funcs.print_scaled_dot_product_attention_out(temp_q, temp_k, temp_v)
        print()

        # This query aligns with a repeated key (third and fourth),
        # so all associated values get averaged.
        temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
        funcs.print_scaled_dot_product_attention_out(temp_q, temp_k, temp_v)
        print()

        # This query aligns equally with the first and second key,
        # so their values get averaged.
        temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)  # (1, 3)
        funcs.print_scaled_dot_product_attention_out(temp_q, temp_k, temp_v)
        print()

        temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype=tf.float32)  # (3, 3)
        funcs.print_scaled_dot_product_attention_out(temp_q, temp_k, temp_v)

        inp = tf.constant([
            [1, 2, 3, 4],
            [3, 4, 5, 6]
        ])
        tar = inp
        enc_padding_mask, combined_mask, dec_padding_mask = funcs.create_masks(inp, tar)
        print(enc_padding_mask.shape)
        print(combined_mask.shape)
        print(dec_padding_mask.shape)


if __name__ == '__main__':
    tf.test.main()
