import tensorflow as tf
from transformers_keras.losses import (MaskedSparseCategoricalCrossentropy,
                                       masked_sparse_categorical_crossentropy)


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
        # the last two time steps is masking positions, so these two losses should be equal
        loss0 = masked_sparse_categorical_crossentropy(x, y0)
        loss1 = masked_sparse_categorical_crossentropy(x, y1)
        self.assertEqual(loss0, loss1)

        # the first 4 positions are the same, so masked ce and ce should be almost equal(masked ce uses epsilon=1e-6)
        ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(x, y0)
        self.assertAlmostEqual(ce.numpy(), loss0.numpy(), delta=6)

        mscc = MaskedSparseCategoricalCrossentropy(from_logits=False, axis=-1)
        _loss0 = mscc(x, y0)
        _loss1 = mscc(x, y1)
        self.assertAlmostEqual(loss0.numpy(), _loss0.numpy(), delta=6)
        self.assertAlmostEqual(loss1.numpy(), _loss1.numpy(), delta=6)


if __name__ == "__main__":
    tf.test.main()
