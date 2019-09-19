import tensorflow as tf


class PointWiseFeedForwardNetwork(tf.keras.Model):

    def __init__(self, d_model, dff):
        super(PointWiseFeedForwardNetwork, self).__init__(name='ffn')
        # self.d_model = d_model
        self.dense_1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(d_model)

    def call(self, x, training=None, mask=None):
        x = self.dense_1(x)
        return self.dense_2(x)

    def compute_output_shape(self, input_shape):
        # shapes = tf.shape(input_shape).as_list()
        # shapes[-1] = self.d_model
        # return tf.TensorShape(shapes)
        return self.dense_2.output_shape
