import unittest

from .bert_tokenizer import BertTokenizer
from .tokenizer import *


class TokenizersTest(unittest.TestCase):

    def testBasicTokenizer(self):
        bt = BasicTokenizer()
        tokens = bt.tokenize('中国的java web工程师')
        self.assertEqual(['中', '国', '的', 'java', 'web', '工', '程', '师'], tokens)

    def testWordPieceTokenizer(self):
        vocab = {
            '##va': 2,
            'web': 3,
            '中国的': 4,
            '工程师': 5
        }
        wpt = WordpieceTokenizer(vocab, unk_token='<UNK>',)
        tokens = wpt.tokenize('中国的java web工程师')
        print(tokens)

    def testBertTokenizer(self):
        tokenizer = BertTokenizer(vocab_file=None)
        print(tokenizer.vocab)


if __name__ == "__main__":
    unittest.main()
