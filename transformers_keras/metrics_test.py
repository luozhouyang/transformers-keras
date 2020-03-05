import tensorflow as tf

from .metrics import masked_sparse_categorical_accuracy


class MetricsTest(tf.test.TestCase):

    def testMaskedSparseCategoricalAccuracy(self):
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
        # the last two time steps is masking positions, so these two accuracy should be equal
        acc0 = masked_sparse_categorical_accuracy(x, y0)
        acc1 = masked_sparse_categorical_accuracy(x, y1)
        self.assertEqual(acc0.numpy(), acc1.numpy())

        # the first 4 positions are the same, so masked acc and acc should be almost equal(masked acc uses epsilon=1e-6)
        acc = tf.keras.metrics.SparseCategoricalAccuracy()(x, y0)
        self.assertAlmostEqual(acc.numpy(), acc0.numpy(), delta=6)


if __name__ == "__main__":
    tf.test.main()
