import abc
import logging
import os

import jieba

from .tokenizer import BasicTokenizer, WordpieceTokenizer


class TransformerAbstractTokenizer(abc.ABC):

    def __init__(self,
                 pad_token='[PAD]',
                 unk_token='[UNK]',
                 sos_token='[SOS]',
                 eos_token='[EOS]',
                 **kwargs):
        super().__init__()
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_id = 0
        self.unk_id = 1
        self.sos_id = 2
        self.eos_id = 3

    def tokenize(self, sequence):
        raise NotImplementedError()

    def encode(self, sequence):
        raise NotImplementedError()

    def decode(self, sequence):
        raise NotImplementedError()

    @property
    def vocab_size(self):
        raise NotImplementedError()

    @property
    def special_tokens(self):
        return [self.pad_token, self.unk_token, self.sos_token, self.eos_token]

    @property
    def special_ids(self):
        return [self.pad_id, self.unk_id, self.sos_id, self.eos_id]

    @property
    def special_token_and_ids(self):
        return [(self.pad_token, self.pad_id),
                (self.unk_token, self.unk_id),
                (self.sos_token, self.sos_id),
                (self.eos_token, self.eos_id)]


class TransformerVocabBasedTokenizer(TransformerAbstractTokenizer):

    def __init__(self, vocab_file, **kwargs):
        super().__init__(**kwargs)
        _sorted_special_tokens = [x[0] for x in sorted(self.special_token_and_ids, key=lambda x:x[1])]
        self.vocab = _sorted_special_tokens + self._read_vocabs(vocab_file)
        self.token2id = self._build_index(self.vocab)
        self.id2token = self._build_reverse_index(self.token2id)

    def _read_vocabs(self, file):
        vocabs = []
        with open(file, mode='rt', encoding='utf8') as fin:
            for line in fin:
                line = line.strip('\n').strip()
                if not line:
                    continue
                if line in self.special_tokens:
                    continue
                vocabs.append(line)
        return vocabs

    def _build_index(self, vocab):
        m = {}
        for i, v in enumerate(vocab):
            m[v] = i
        return m

    def _build_reverse_index(self, token2id):
        m = {}
        for k, v in token2id.items():
            m[v] = k
        return m

    def tokenize(self, sequence):
        raise NotImplementedError()

    def encode(self, sequence):
        tokens = self.tokenize(sequence)
        return [self.token2id.get(t, self.unk_id) for t in tokens]

    def decode(self, sequence):
        return [self.id2token.get(_id, self.unk_token) for _id in sequence]

    @property
    def vocab_size(self):
        return len(self.token2id)


class TransformerDefaultTokenizer(TransformerVocabBasedTokenizer):

    def __init__(self,
                 vocab_file,
                 do_basic_tokenization=True,
                 do_lower_case=True,
                 nerver_split=None,
                 tokenize_chinese_chars=True,
                 max_input_chars_per_word=100,
                 **kwargs):
        super(TransformerDefaultTokenizer, self).__init__(vocab_file, **kwargs)
        self.do_basic_tokenization = do_basic_tokenization
        self.do_lower_case = do_lower_case
        self.never_split = nerver_split
        self.tokenize_chinese_chars = tokenize_chinese_chars
        if self.do_basic_tokenization:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=self.do_lower_case,
                never_split=self.never_split,
                tokenize_chinese_chars=self.tokenize_chinese_chars
            )
        else:
            self.basic_tokenizer = None

        self.max_input_chars_per_word = kwargs.get('max_input_chars_per_word', 100)
        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.vocab,
            unk_token=self.unk_token,
            max_input_chars_per_word=self.max_input_chars_per_word
        )

    def tokenize(self, sequence):
        tokens = []
        if self.do_basic_tokenization:
            for token in self.basic_tokenizer.tokenize(sequence, never_split=self.never_split):
                for t in self.wordpiece_tokenizer.tokenize(token):
                    tokens.append(t)
        else:
            tokens = self.wordpiece_tokenizer.tokenize(sequence)
        return tokens


class TransformerJiebaTokenizer(TransformerVocabBasedTokenizer):

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
