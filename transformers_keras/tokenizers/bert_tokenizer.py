from .tokenizer import BasicTokenizer, WordpieceTokenizer


class BertTokenizer:

    def __init__(self,
                 vocab_file,
                 pad_token='[PAD]',
                 unk_token='[UNK]',
                 cls_token='[CLS]',
                 sep_token='[SEP]',
                 mask_token='[MASK]',
                 do_lower_case=True,
                 do_basic_tokenization=True,
                 tokenize_chinese_chars=True,
                 never_split=None,
                 max_input_chars_per_word=100,
                 **kwargs):
        super().__init__()
        # load vocab dict from file
        self.vocab = self._load_vocab(vocab_file)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        assert len(self.vocab) == len(self.reverse_vocab)

        self.vocab_size = len(self.vocab)
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token
        self.pad_id = self.vocab[self.pad_token]
        self.unk_id = self.vocab[self.unk_token]
        self.cls_id = self.vocab[self.cls_token]
        self.sep_id = self.vocab[self.sep_token]
        self.mask_id = self.vocab[self.mask_token]

        self.never_split = never_split or []
        self.do_basic_tokenization = do_basic_tokenization

        if self.do_basic_tokenization:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars)

        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=set(self.vocab.keys()),
            unk_token=self.unk_token,
            max_input_chars_per_word=max_input_chars_per_word)

    def tokenize(self, text, never_split=None, **kwargs):
        if not self.do_basic_tokenization:
            return self.wordpiece_tokenizer.tokenize(text)
        tokens = []
        never_split = never_split + self.never_split if never_split is not None else self.never_split
        for token in self.basic_tokenizer.tokenize(text, never_split=never_split):
            for t in self.wordpiece_tokenizer.tokenize(token):
                tokens.append(t)
        return tokens

    def encode(self, text, add_cls=True, add_sep=True, never_split=None, **kwargs):
        ids = []
        for token in self.tokenize(text, never_split=never_split, **kwargs):
            ids.append(self.vocab.get(token, self.unk_id))
        if add_cls:
            ids = [self.cls_id] + ids
        if add_sep:
            ids = ids + [self.sep_id]
        return ids

    def decode(self, ids, drop_cls=True, drop_sep=True, **kwargs):
        tokens = [self.reverse_vocab.get(_id, self.unk_token) for _id in ids]
        if drop_cls and tokens[0] == self.cls_token:
            tokens = tokens[1:]
        if drop_sep and tokens[-1] == self.sep_token:
            tokens = tokens[:-1]
        return tokens

    def _load_vocab(self, vocab_file):
        words = []
        with open(vocab_file, mode='rt', encoding='utf-8') as fin:
            for line in fin:
                word = line.rstrip('\n')
                words.append(word)
        vocab = {}
        for idx, word in enumerate(words):
            vocab[word] = idx
        return vocab
