import tensorflow as tf
from transformers_keras.transformer.multi_head_attention import MultiHeadAttention


class MultiHeadAttentionTest(tf.test.TestCase):

    def testMultiHeadAttention(self):
        mha = MultiHeadAttention(512, 8)
        y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
        out, attn = mha((y, y, y), True, mask=None)
        self.assertAllEqual(out.shape, [1, 60, 512])
        self.assertAllEqual(attn.shape, [1, 8, 60, 60])


if __name__ == '__main__':
    tf.test.main()
