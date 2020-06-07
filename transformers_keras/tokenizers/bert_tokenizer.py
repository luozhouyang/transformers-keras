import abc

from .tokenizer import BasicTokenizer, WordpieceTokenizer


class BertAbstractTokenizer(abc.ABC):

    def __init__(self,
                 pad_token='[PAD]',
                 unk_token='[UNK]',
                 sos_token='[SOS]',
                 eos_token='[EOS]',
                 cls_token='[CLS]',
                 sep_token='[SEP]',
                 mask_token='[MASK]',
                 **kwargs):
        super().__init__()
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token

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
        return [self.pad_token, self.unk_token, self.sos_token, self.eos_token,
                self.cls_token, self.sep_token, self.mask_token]


class BertVocabBasedTokenizer(BertAbstractTokenizer):
    """Tokenizer that based on a vocab file. All special tokens will be moved to the head of the file."""

    def __init__(self, vocab_file, **kwargs):
        super(BertVocabBasedTokenizer, self).__init__(**kwargs)
        # vocabs = special tokens + vocab tokens, move special tokens to the head of list
        self.vocab = self.special_tokens + self._load_vocab(vocab_file)
        self.token2id = self._build_index(self.vocab)
        self.id2token = self._build_reverse_index(self.token2id)
        self.pad_id = self.token2id[self.pad_token]
        self.unk_id = self.token2id[self.unk_token]
        self.sos_id = self.token2id[self.sos_token]
        self.eos_id = self.token2id[self.eos_token]
        self.cls_id = self.token2id[self.cls_token]
        self.sep_id = self.token2id[self.sep_token]
        self.mask_id = self.token2id[self.mask_token]

    def _load_vocab(self, file):
        vocabs = []
        with open(file, mode='rt', encoding='utf8') as fin:
            for line in fin:
                word = line.strip('\n').strip()
                if not word:
                    continue
                if word in self.special_tokens:
                    continue
                vocabs.append(word)
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

    @property
    def vocab_size(self):
        return len(self.token2id)

    def tokenize(self, sequence):
        raise NotImplementedError()

    def encode(self, sequence):
        tokens = self.tokenize(sequence)
        return [self.token2id.get(t, self.unk_id) for t in tokens]

    def decode(self, sequence):
        return [self.id2token.get(_id, self.unk_id) for _id in sequence]


class BertDefaultTokenizer(BertVocabBasedTokenizer):
    """Default tokenizer, support basic tokenizer and wordpiece tokenizer."""

    def __init__(self, vocab_file, **kwargs):
        super().__init__(vocab_file, **kwargs)
        self.do_basic_tokenization = kwargs.get('do_basic_tokenization', True)
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
