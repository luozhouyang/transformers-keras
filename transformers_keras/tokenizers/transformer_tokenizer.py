import os

from .abstract_tokenizer import AbstractTokenizerV2
from .tokenizer import BasicTokenizer, WordpieceTokenizer


class TransformerTokenizer(AbstractTokenizerV2):

    def __init__(self, vocab_file, **kwagrs):
        super().__init__(vocab_file, **kwagrs)

        self._do_basic_tokenization = kwagrs.get('do_basic_tokenization', True)
        if self._do_basic_tokenization:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=kwagrs.get('do_lower_case', True),
                never_split=kwagrs.get('never_split', None),
                tokenize_chinese_chars=kwagrs.get('tokenize_chinese_chars'))
        else:
            self.basic_tokenizer = None

        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self._vocab,
            unk_token=self._unk_token,
            max_input_chars_per_word=kwagrs.get('max_input_chars_per_word', 100))

    def tokenize(self, text, *inputs, **kwargs):
        tokens = []
        if self._do_basic_tokenizer:
            for token in self.basic_tokenizer.tokenize(text, never_split=kwargs.get('never_split', None)):
                for t in self.wordpiece_tokenizer.tokenize(token):
                    tokens.append(t)
        else:
            tokens = self.wordpiece_tokenizer.tokenize(text)
        return token
