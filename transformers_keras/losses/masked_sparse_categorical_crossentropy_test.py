import tensorflow as tf

from .masked_sparse_categorical_crossentropy import \
    MaskedSparseCategoricalCrossentropy


class MaskedSparseCategoricalCrossentropyTest(tf.test.TestCase):

    def testMaskedSparseCategoricalCrossentropy(self):
        ce = MaskedSparseCategoricalCrossentropy(from_logits=True, mask_id=5)
        x = tf.constant([1, 2, 3, 4, 5, 6], shape=(2, 3), dtype=tf.int32)
        y = tf.random.uniform(shape=(2, 3, 8), dtype=tf.float32)
        y = tf.nn.softmax(y)
        loss = ce(x, y)
        print(loss.numpy())


if __name__ == "__main__":
    tf.test.main()
