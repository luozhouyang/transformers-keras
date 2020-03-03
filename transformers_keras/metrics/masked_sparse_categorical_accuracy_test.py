import tensorflow as tf

from .masked_sparse_categorical_accuracy import MaskedSparseCategoricalAccuracy


class MaskedSparseCategoricalAccuracyTest(tf.test.TestCase):

    def testMaskedSparseCategoricalAccuracy(self):
        acc = MaskedSparseCategoricalAccuracy(name='acc')

        x = tf.constant([1, 2, 3, 4, 5, 6], shape=(2, 3), dtype=tf.int32)
        y = tf.random.uniform(shape=(2, 3, 8), dtype=tf.float32)
        y = tf.nn.softmax(y)

        acc.update_state(x, y)
        print('acc result: ', acc.result().numpy())
        print(acc(x, y).numpy())


if __name__ == "__main__":
    tf.test.main()
