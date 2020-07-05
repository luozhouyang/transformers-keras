import unittest

import tensorflow as tf

from .albert_adapter import AlbertAdapter, ChineseAlbertLargeStrategy


class AlbertAdapterTest(unittest.TestCase):

    def testAlbertAdapter(self):
        adapter = AlbertAdapter('zh-albert-large')
        model, vocab = adapter.adapte('/Users/luozhouyang/pretrain-models/albert/albert_large')


if __name__ == "__main__":
    unittest.main()
