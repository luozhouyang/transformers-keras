import tensorflow as tf

from .masked_sparse_categorical_accuracy import MaskedSparseCategoricalAccuracy


class MaskedSparseCategoricalAccuracyTest(tf.test.TestCase):

    def testMaskedSparseCategoricalAccuracy(self):
        acc = MaskedSparseCategoricalAccuracy(name='acc')

        x = tf.constant([1, 2, 3, 4, 0, 0], shape=(2, 3), dtype=tf.int32)
        y = tf.constant([
            [[.5, .0, .0, .0, .1, .2, .1, .1], [.0, .0, .5, .0, .1, .2, .1, .1], [.0, .0, .0, .5, .1, .2, .1, .1]],
            [[.0, .1, .0, .0, .5, .2, .1, .1], [.0, .5, .0, .0, .1, .2, .1, .1], [.0, .5, .0, .0, .1, .2, .1, .1]]
        ], shape=(2, 3, 8), dtype=tf.float32)

        acc.update_state(x, y)
        self.assertEquals(acc.result().numpy(), 3.0/4)

        y = tf.constant([
            [[.5, .0, .0, .0, .1, .2, .1, .1], [.0, .5, .0, .0, .1, .2, .1, .1], [.0, .0, .0, .5, .1, .2, .1, .1]],
            [[.0, .1, .0, .0, .5, .2, .1, .1], [.0, .5, .0, .0, .1, .2, .1, .1], [.0, .5, .0, .0, .1, .2, .1, .1]]
        ], shape=(2, 3, 8), dtype=tf.float32)
        acc.update_state(x, y)
        self.assertEquals(acc.result().numpy(), 5.0/8)


if __name__ == "__main__":
    tf.test.main()
