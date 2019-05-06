import tensorflow as tf
from transformer.layer_norm import LayerNormalization


class LayerNormalizationTest(tf.test.TestCase):

    def testLayerNormalizationLayer(self):
        _ = self
        ln = LayerNormalization(1e-6)
        x = tf.random.uniform((64, 48, 512))
        output = ln(x)
        print(output)


if __name__ == '__main__':
    tf.test.main()
