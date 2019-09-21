import os
import unittest
import tensorflow as tf

from transformers_keras.tokenizers.space_tokenizer import SpaceTokenizer


class SpaceTokenizerTest(unittest.TestCase):

    def buildTokenizer(self):
        tokenizer = SpaceTokenizer()
        corpus = ['train.tgt.txt']
        corpus = [os.path.join('testdata', f) for f in corpus]
        tokenizer.tokenize(corpus)
        return tokenizer

    def testTokenize(self):
        tokenizer = self.buildTokenizer()
        print(tokenizer.tokens2ids_dict)
        print(tokenizer.ids2tokens_dict)
        print(tokenizer.vocab_size)

    def testConvertTokens2Ids(self):
        tokenizer = self.buildTokenizer()
        tokenizer.initialize_lookup_tables()
        print('token2 id vocab: ', tokenizer.tokens2ids_dict)
        words = tf.constant(['I', 'am', 'a', 'developer'])
        v = tokenizer.tokens2ids(words)
        print(v)

    def testConvertIds2Tokens(self):
        tokenizer = self.buildTokenizer()
        tokenizer.initialize_lookup_tables()
        print('id2token vocab: ', tokenizer.ids2tokens_dict)
        ids = tf.constant([1, 0, 2, 3, 4], dtype=tf.dtypes.int64)
        v = tokenizer.ids2tokens(ids)
        print(v)


if __name__ == '__main__':
    unittest.main()
