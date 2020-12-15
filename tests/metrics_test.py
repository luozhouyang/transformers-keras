import tensorflow as tf
from transformers_keras.metrics import (MaskedSparseCategoricalAccuracy,
                                        masked_sparse_categorical_accuracy)


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
        self.assertAlmostEqual(1.0, acc0, delta=6)
        self.assertAlmostEqual(1.0, acc1, delta=6)

        sca = tf.keras.metrics.SparseCategoricalAccuracy()
        acc = sca(x, y0)
        self.assertAlmostEqual(acc.numpy(), acc0.numpy(), delta=6)
        self.assertAlmostEqual(1.0, acc, delta=6)

        msca = MaskedSparseCategoricalAccuracy(mask_id=0, from_logits=False)
        _acc0 = msca(x, y0)
        self.assertEqual(4.0, msca.total)
        self.assertEqual(4.0, msca.count)
        self.assertEqual(4.0 / 4.0, _acc0)

        _acc1 = msca(x, y1)
        self.assertEqual(8.0, msca.total)
        self.assertEqual(8.0, msca.count)
        self.assertEqual(8.0 / 8.0, _acc1)

        y2 = tf.constant([
            [
                [.1, .5, .2, .1, .1],  # 1
                [.1, .2, .5, .1, .1],  # 2
                [.1, .2, .1, .5, .1],  # 3
            ],
            [
                [.1, .1, .5, .2, .1],  # 4
                [.1, .1, .1, .2, .5],  # 4, masked
                [.1, .1, .5, .2, .1],  # 2, masked
            ]],
            shape=(2, 3, 5),
            dtype=tf.float32
        )
        acc2 = masked_sparse_categorical_accuracy(x, y2)
        _acc2 = msca(x, y2)

        self.assertAlmostEqual(0.75, acc2, delta=6)
        self.assertEqual(11.0, msca.total)
        self.assertEqual(12.0, msca.count)
        self.assertAlmostEqual(11.0 / 12.0, _acc2, delta=6)

        print(' acc0: ', acc0.numpy())
        print(' acc1: ', acc1.numpy())
        print(' acc2: ', acc2.numpy())
        print('_acc0: ', _acc0.numpy())
        print('_acc1: ', _acc1.numpy())
        print('_acc2: ', _acc2.numpy())


if __name__ == "__main__":
    tf.test.main()
