import logging

import jieba

from .transformer_tokenizer import TransformerVocabBasedTokenizer


class TransformerChineseJiebaTokenizer(TransformerVocabBasedTokenizer):

    def __init__(self, vocab_file, jieba_userdict=None, **kwargs):
        super().__init__(vocab_file, **kwargs)
        if jieba_userdict:
            jieba.load_userdict(jieba_userdict)
            logging.info('jieba load user dict: %s finished.' % jieba_userdict)

    def tokenize(self, sequence):
        tokens = []
        for t in jieba.cut(sequence):
            w = t.strip()
            if not w:
                continue
            tokens.append(w)
        return tokens
