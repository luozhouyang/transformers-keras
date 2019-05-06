import tensorflow as tf
from transformer.multi_head_attention import MultiHeadAttention


class MultiHeadAttentionTest(tf.test.TestCase):

    def testMultiHeadAttention(self):
        mha = MultiHeadAttention(512, 8)
        y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
        out, attn = mha(y, k=y, q=y, mask=None)
        print(out.shape)
        print(attn.shape)


if __name__ == '__main__':
    tf.test.main()
