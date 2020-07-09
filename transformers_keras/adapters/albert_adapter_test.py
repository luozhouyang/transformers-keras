import unittest

import tensorflow as tf

from .albert_adapter import AlbertAdapter, ChineseAlbertLargeStrategy


class AlbertAdapterTest(unittest.TestCase):

    def testAlbertAdapter(self):
        adapter = AlbertAdapter('zh-albert-large')
        model, vocab = adapter.adapte('/Users/luozhouyang/pretrain-models/albert/albert_large')

        input_ids = [0, 1, 2, 3, 4, 5, 6, 0, 0, 0]
        input_ids = input_ids + [0] * (512 - len(input_ids))
        input_ids = tf.constant(input_ids, dtype=tf.int32, shape=(1, 512))
        input_mask = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        input_mask = input_mask + [1] * (512 - len(input_mask))
        input_mask = tf.constant(input_mask, dtype=tf.float32, shape=(1, 512))
        segment_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        segment_ids = segment_ids + [0] * (512 - len(segment_ids))
        segment_ids = tf.constant(segment_ids, dtype=tf.int32, shape=(1, 512))

        outputs = model.predict(x=(input_ids, input_mask, segment_ids))

        print(outputs[0])
        print(outputs[1])

        outputs = model(inputs=(input_ids, input_mask, segment_ids))
        print(outputs[0].numpy())
        print(outputs[1].numpy())


if __name__ == "__main__":
    unittest.main()
