import tensorflow as tf

from .losses import masked_sparse_categorical_crossentropy


class LossesTest(tf.test.TestCase):

    def testMaskedSparseCategoricalCrossentropy(self):
        x = tf.constant([1, 2, 3, 4, 0, 0], shape=(2, 3), dtype=tf.int32)
        y0 = tf.constant([
            [
                [.1, .5, .2, .1, .1],  # 1
                [.1, .2, .5, .1, .1],  # 2
                [.1, .2, .1, .5, .1],  # 3
            ],
            [
                [.1, .1, .1, .2, .5],  # 4
                [.5, .1, .1, .2, .1],  # 0, masked
                [.5, .1, .1, .2, .1],  # 0, masked
            ]],
            shape=(2, 3, 5),
            dtype=tf.float32
        )
        y1 = tf.constant([
            [
                [.1, .5, .2, .1, .1],  # 1
                [.1, .2, .5, .1, .1],  # 2
                [.1, .2, .1, .5, .1],  # 3
            ],
            [
                [.1, .1, .1, .2, .5],  # 4
                [.1, .1, .1, .2, .5],  # 4, masked
                [.1, .1, .5, .2, .1],  # 2, masked
            ]],
            shape=(2, 3, 5),
            dtype=tf.float32
        )
        # the last two time steps is masking positions, so these tow losses should be equal
        loss0 = masked_sparse_categorical_crossentropy(x, y0)
        loss1 = masked_sparse_categorical_crossentropy(x, y1)
        self.assertEqual(loss0, loss1)


if __name__ == "__main__":
    tf.test.main()
