import tensorflow as tf

from transformer.point_wise_ffn import PointWiseFeedForwardNetwork


class PointWiseFeedForwardNetworkTest(tf.test.TestCase):

    def testPointWiseFFN(self):
        ffn = PointWiseFeedForwardNetwork(512, 2048)
        output = ffn(tf.random.uniform((64, 50, 256)))
        print(output.shape)


if __name__ == '__main__':
    tf.test.main()
