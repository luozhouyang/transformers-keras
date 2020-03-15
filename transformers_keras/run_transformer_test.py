import tensorflow as tf

from .modeling_transformer import *
from .run_transformer import *


class TransformerTest(tf.test.TestCase):

    def testTransformerTraining(self):
        config = TransformerConfig(max_positions=10)
        model = build_model(config)
        model.summary()

        x = tf.constant([1, 2, 3, 4, 5, 6, 0, 0, 0, 0], shape=(1, 10), dtype=tf.int32)
        y = tf.constant([2, 3, 4, 5, 6, 1, 7, 8, 0, 0], shape=(1, 10), dtype=tf.int32)
        inputs = {
            'x': x,
            'y': y,
        }
        labels = {
            'probs': y,
        }
        input_dataset = tf.data.Dataset.from_tensor_slices(inputs)
        label_dataset = tf.data.Dataset.from_tensor_slices(labels)
        dataset = tf.data.Dataset.zip((input_dataset, label_dataset))
        dataset = dataset.repeat(100).batch(2)
        print(next(iter(dataset)))

        model.fit(dataset, epochs=4)

        model.save_weights('/tmp/transformer_weights', save_format='tf')

    def testLoadPretrainedModel(self):
        config = TransformerConfig(max_positions=10)
        model = build_model(config)

        model.load_weights('/tmp/transformer_weights')

        x = tf.constant([1, 2, 3, 4, 5, 6, 0, 0, 0, 0], shape=(1, 10), dtype=tf.int32)
        y = tf.constant([2, 3, 4, 5, 6, 1, 7, 8, 0, 0], shape=(1, 10), dtype=tf.int32)
        inputs = {
            'x': x,
            'y': y,
        }
        labels = {
            'probs': y,
        }
        input_dataset = tf.data.Dataset.from_tensor_slices(inputs)
        label_dataset = tf.data.Dataset.from_tensor_slices(labels)
        dataset = tf.data.Dataset.zip((input_dataset, label_dataset))
        dataset = dataset.repeat(100).batch(2)
        # NOTE: this is not the prediction that we generate an output sequence by source inputs
        probs = model.predict(dataset)
        print(probs)


if __name__ == "__main__":
    tf.test.main()
