import abc
import os

from .abstract_tokenizer import AbstractTokenizerV2
from .tokenizer import BasicTokenizer, WordpieceTokenizer


class TransformerAbstractTokenizer(abc.ABC):

    def __init__(self, **kwargs):
        super().__init__()
        self.pad_token = kwargs.get('pad_token', '<PAD>')
        self.unk_token = kwargs.get('unk_token', '<UNK>')
        self.sos_token = kwargs.get('sos_token', '<S>')
        self.eos_token = kwargs.get('eos_token', '</S>')
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


class TransformerDefaultTokenizer(TransformerVocabBasedTokenizer):

    def __init__(self, vocab_file, **kwargs):
        super(TransformerDefaultTokenizer, self).__init__(vocab_file, **kwargs)
        self.do_basic_tokenization = kwargs.get('do_basic_tokenization')
        self.do_lower_case = kwargs.get('do_lower_case', True)
        self.never_split = kwargs.get('never_split', None)
        self.tokenize_chinese_chars = kwargs.get('tokenize_chinese_chars', True)
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
