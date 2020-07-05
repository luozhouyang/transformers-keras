import unittest

from .bert_adapter import BertAdapter


class BertAdapterTest(unittest.TestCase):

    def testLoadPretrainedBertModel(self):
        adapter = BertAdapter(strategy='chinese-bert-base')
        model, vocab_file = adapter.adapte('/Users/luozhouyang/pretrain-models/bert/chinese_L-12_H-768_A-12')

        print('model inputs: {}'.format(model.inputs))
        print('model outputs: {}'.format(model.outputs))


if __name__ == "__main__":
    unittest.main()
