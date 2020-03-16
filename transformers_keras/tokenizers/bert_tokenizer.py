import logging

from .abstract_tokenizer import AbstractTokenizerV2
from .tokenizer import BasicTokenizer, WordpieceTokenizer


class BertTokenizer(AbstractTokenizerV2):

    def __init__(self, vocab_file, **kwagrs):
        super().__init__(vocab_file, **kwagrs)

        cls_token = kwagrs.get('cls_token', '[CLS]')
        if cls_token not in self._vocab:
            logging.info('Vocab file not contains an `cls_token`, set to %s, id is: %d' % (cls_token, self._id))
            self._vocab[cls_token] = self._id
            self._id += 1
        self._cls_token = cls_token

        sep_token = kwagrs.get('sep_token', '[SEP]')
        if sep_token not in self._vocab:
            logging.info('Vocab file not contains an `sep_token`, set to %s, id is: %d' % (sep_token, self._id))
            self._vocab[sep_token] = self._id
            self._id += 1
        self._sep_token = sep_token

        mask_token = kwagrs.get('mask_token', '[MASK]')
        if mask_token not in self._vocab:
            logging.info('Vocab file not contains an `mask_token`, set to %s, id is: %d' % (mask_token, self._id))
            self._vocab[mask_token] = self._id
            self._id += 1
        self._mask_token = mask_token

        self._build_reverse_vocab()

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

    @property
    def cls_token(self):
        return self._cls_token

    @property
    def cls_token_id(self):
        return self._vocab[self.cls_token]

    @property
    def sep_token(self):
        return self._sep_token

    @property
    def sep_token_id(self):
        return self._vocab[self.sep_token]

    @property
    def mask_token(self):
        return self._mask_token

    @property
    def mask_token_id(self):
        return self._vocab[self.mask_token]
