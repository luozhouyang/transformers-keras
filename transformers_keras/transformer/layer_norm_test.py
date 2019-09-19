import tensorflow as tf
from transformers_keras.transformer.layer_norm import LayerNormalization
from tensorflow.python import keras


class LayerNormalizationTest(tf.test.TestCase):

    def testLayerNormalizationLayer(self):
        _ = self
        ln = LayerNormalization(1e-6)
        x = tf.random.uniform((64, 48, 512))
        output = ln(x)
        print(output)

        ln = keras.layers.LayerNormalization(norm_axis=-1)
        output2 = ln(x)

        self.assertAlmostEqual(output, output2)


if __name__ == '__main__':
    tf.test.main()
