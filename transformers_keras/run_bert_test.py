import numpy as np
import tensorflow as tf

from .modeling_bert import *
from .run_bert import build_model


class BertTest(tf.test.TestCase):

    def testBert4PretrainingModel(self):
        config = BertConfig(vocab_size=15, max_sequence_length=8)
        model = build_model(config)

        inputs = {
            'input_ids':  tf.constant([1, 2, 3, 4, 5, 6, 7, 8], shape=(1, 8), dtype=tf.int32),
            'input_mask': tf.constant([1, 1, 1, 1, 1, 1, 1, 0],  shape=(1, 8), dtype=tf.int32),
            'segment_ids':  tf.constant([0, 0, 0, 0, 1, 1, 1, 1], shape=(1, 8),  dtype=tf.int32)
        }

        labels = {
            'relations': tf.one_hot(tf.constant([1], shape=(1,),  dtype=tf.int32), 2),
            'predictions': np.random.randint(low=0, high=14, size=(1, 8,))
        }

        x_dataset = tf.data.Dataset.from_tensor_slices(inputs)
        y_dataset = tf.data.Dataset.from_tensor_slices(labels)
        dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
        train_dataset = dataset.repeat(50).batch(2, drop_remainder=True)
        print(next(iter(train_dataset)))
        model.fit(train_dataset, validation_data=train_dataset, epochs=2)

        # model.save('/tmp/keras_bert_example', include_optimizer=False, save_format='tf')
        # tf.saved_model.save(model, '/tmp/keras_bert_example')
        model.save_weights('/tmp/bert_weights', save_format='tf')

    def testLoadPretrainedModel(self):
        inputs = {
            'input_ids':  tf.constant([1, 2, 3, 4, 5, 6, 7, 8], shape=(1, 8), dtype=tf.int32),
            'input_mask': tf.constant([1, 1, 1, 1, 1, 1, 1, 0],  shape=(1, 8), dtype=tf.int32),
            'segment_ids':  tf.constant([0, 0, 0, 0, 1, 1, 1, 1], shape=(1, 8),  dtype=tf.int32)
        }
        x_dataset = tf.data.Dataset.from_tensor_slices(inputs)
        config = BertConfig(vocab_size=15, max_sequence_length=8)
        new_model = build_model(config)
        new_model.load_weights('/tmp/bert_weights')
        predictions, relations, attentions = new_model.predict(x_dataset.batch(2))
        print(predictions)
        print('='*100)
        print(relations)
        print('='*100)
        print(attentions)


if __name__ == "__main__":
    tf.test.main()
