import os
import unittest

import tensorflow as tf

from transformers_keras.tokenizers.space_tokenizer import SpaceTokenizer


class SpaceTokenizerTest(unittest.TestCase):

    def buildTokenizer(self):
        tokenizer = SpaceTokenizer()
        corpus = ['train.tgt.txt']
        corpus = [os.path.join('testdata', f) for f in corpus]
        tokenizer.build_from_corpus(corpus)
        return tokenizer

    def testTokenize(self):
        tokenizer = self.buildTokenizer()
        print(tokenizer.token2id_dict)
        print(tokenizer.id2token_dict)
        print(tokenizer.vocab_size)

    def testConvertTokens2Ids(self):
        tokenizer = self.buildTokenizer()
        print('token2 id dict: ', tokenizer.token2id_dict)
        words = tf.constant(['I', 'am', 'a', 'developer'])
        v = tokenizer.encode(words)
        print(v)

    def testConvertIds2Tokens(self):
        tokenizer = self.buildTokenizer()
        print('id2token dict: ', tokenizer.id2token_dict)
        ids = tf.constant([1, 0, 2, 3, 4], dtype=tf.dtypes.int64)
        v = tokenizer.decode(ids)
        print(v)

    def testSaveVocabFile(self):
        tokenizer = self.buildTokenizer()
        tokenizer.save_to_vocab('testdata/vocab.test.txt')

    def testBuildFromVocab(self):
        print('============start build from vocab=============')
        tokenizer = SpaceTokenizer()
        tokenizer.build_from_vocab('testdata/vocab.test.txt')
        print('token2id dict: ', tokenizer.token2id_dict)
        print('id2token dict: ', tokenizer.id2token_dict)
        words = tf.constant(['I', 'am', 'a', 'developer'])
        v0 = tokenizer.encode(words)
        print(v0)
        ids = tf.constant([1, 0, 2, 3, 4], dtype=tf.dtypes.int64)
        v1 = tokenizer.decode(ids)
        print(v1)
        print('============end build from vocab=============')


if __name__ == '__main__':
    unittest.main()
