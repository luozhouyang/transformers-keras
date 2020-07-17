import numpy as np
import tensorflow as tf
import os

from .modeling_albert import AlbertForPretrainingModel, AlbertModel
from .modeling_bert import BertForPretrainingModel, BertModel


class PretrainedModelTest(tf.test.TestCase):

    def testBertModelFromPretrained(self):
        model = BertModel.from_pretrained(
            os.path.join(os.path.expanduser('~'), '.transformers_keras/pretrain-models/bert/chinese_L-12_H-768_A-12'))

        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        input_ids = [0, 1, 2, 3, 4, 5, 6, 0, 0, 0]
        input_ids = input_ids + [0] * (512 - len(input_ids))
        input_ids = tf.constant(input_ids, dtype=tf.int32, shape=(1, 512))
        input_mask = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        input_mask = input_mask + [1] * (512 - len(input_mask))
        input_mask = tf.constant(input_mask, dtype=tf.int32, shape=(1, 512))
        segment_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        segment_ids = segment_ids + [0] * (512 - len(segment_ids))
        segment_ids = tf.constant(segment_ids, dtype=tf.int32, shape=(1, 512))

        sequence_outputs, pooled_outputs = model(inputs=(input_ids, input_mask, segment_ids))
        self.assertEqual([1, 512, 768], sequence_outputs.shape)
        self.assertEqual([1, 768], pooled_outputs.shape)
        print(sequence_outputs)
        print(pooled_outputs)

    def testBertFroPretrainingModelFromPretrained(self):
        model = BertForPretrainingModel.from_pretrained(
            os.path.join(os.path.expanduser('~'), '.transformers_keras/pretrain-models/bert/chinese_L-12_H-768_A-12'))

        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        input_ids = [0, 1, 2, 3, 4, 5, 6, 0, 0, 0]
        input_ids = input_ids + [0] * (512 - len(input_ids))
        input_ids = tf.constant(input_ids, dtype=tf.int32, shape=(1, 512))
        input_mask = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        input_mask = input_mask + [1] * (512 - len(input_mask))
        input_mask = tf.constant(input_mask, dtype=tf.int32, shape=(1, 512))
        segment_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        segment_ids = segment_ids + [0] * (512 - len(segment_ids))
        segment_ids = tf.constant(segment_ids, dtype=tf.int32, shape=(1, 512))

        predictions, relations = model(inputs=(input_ids, input_mask, segment_ids))
        self.assertEqual([1, 512, 21128], predictions.shape)
        self.assertEqual([1, 2], relations.shape)
        print(predictions)
        print(relations)

    def testAlbertModelFromPretrained(self):
        input_ids = tf.constant(
            [1, 2, 3, 4, 5, 6, 7, 5, 3, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5, 6, 6, 6, 7, 7, 8, 0, 0, 0, 0, 0, 0],
            shape=(2, 16),
            dtype=np.int32)  # input_ids
        token_type_ids = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int64)  # token_type_ids,
        input_mask = tf.constant(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]], dtype=np.float32)  # input_mask

        model = AlbertModel.from_pretrained(
            os.path.join(os.path.expanduser('~'), '.transformers_keras/pretrain-models/albert/albert_large'))
        outputs, pooled_outputs = model(inputs=(input_ids, token_type_ids, input_mask))
        self.assertAllEqual([2, 16, 1024], outputs.shape)
        self.assertAllEqual([2, 1024], pooled_outputs.shape)

    def testAlbertForPretrainingModelFromPretrained(self):
        input_ids = tf.constant(
            [1, 2, 3, 4, 5, 6, 7, 5, 3, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5, 6, 6, 6, 7, 7, 8, 0, 0, 0, 0, 0, 0],
            shape=(2, 16),
            dtype=np.int32)  # input_ids
        token_type_ids = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int64)  # token_type_ids,
        input_mask = tf.constant(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]], dtype=np.float32)  # input_mask

        model = AlbertForPretrainingModel.from_pretrained(
            os.path.join(os.path.expanduser('~'), '.transformers_keras/pretrain-models/albert/albert_large'))
        outputs, pooled_outputs = model(inputs=(input_ids, token_type_ids, input_mask))
        self.assertAllEqual([2, 16, 21128], outputs.shape)
        self.assertAllEqual([2, 2], pooled_outputs.shape)


if __name__ == "__main__":
    tf.test.main()
