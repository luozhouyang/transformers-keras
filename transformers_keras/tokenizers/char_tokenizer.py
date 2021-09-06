from collections import namedtuple

TokenizerEncoding = namedtuple("TokenizerEncoding", ["text", "tokens", "ids", "type_ids", "attention_mask", "offsets"])


class BertCharTokenizer:
    """Bert char level tokenizer for token classification tasks."""

    def __init__(
        self,
        token2id,
        do_lower_case=True,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        **kwargs
    ):
        super().__init__()
        self.token2id = token2id
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token
        self.unk_id = self.token2id[self.unk_token]
        self.pad_id = self.token2id[self.pad_token]
        self.cls_id = self.token2id[self.cls_token]
        self.sep_id = self.token2id[self.sep_token]
        self.mask_id = self.token2id[self.mask_token]
        self.do_lower_case = do_lower_case

    @classmethod
    def from_file(cls, vocab_file, **kwargs):
        token2id = cls._load_vocab(vocab_file=vocab_file)
        tokenizer = cls(token2id, **kwargs)
        return tokenizer

    @classmethod
    def _load_vocab(cls, vocab_file):
        vocab = {}
        idx = 0
        with open(vocab_file, mode="rt", encoding="utf-8") as fin:
            for line in fin:
                word = line.rstrip("\n")
                vocab[word] = idx
                idx += 1
        return vocab

    def encode(self, text, add_cls=True, add_sep=True, **kwargs):
        tokens, ids, type_ids, attention_mask, offsets = [], [], [], [], []
        if self.do_lower_case:
            text = str(text).lower()
        for idx, char in enumerate(text):
            tokens.append(char)
            ids.append(self.token2id.get(char, self.unk_id))
            type_ids.append(0)
            attention_mask.append(1)
            offsets.append((idx, idx + 1))

        if add_cls:
            tokens.insert(0, self.cls_token)
            ids.insert(0, self.cls_id)
            type_ids.insert(0, 0)
            attention_mask.insert(0, 1)
            offsets.insert(0, (0, 0))

        if add_sep:
            tokens.append(self.sep_token)
            ids.append(self.sep_id)
            type_ids.append(0)
            attention_mask.append(1)
            offsets.append((0, 0))

        encoding = TokenizerEncoding(
            text=text,
            tokens=tokens,
            ids=ids,
            type_ids=type_ids,
            attention_mask=attention_mask,
            offsets=offsets,
        )
        return encoding

    def decode(self, ids, **kwargs):
        tokens = [self.id2token.get(_id, self.unk_token) for _id in ids]
        return tokens

    def token_to_id(self, token, **kwargs):
        if self.do_lower_case:
            token = token.lower()
        return self.token2id.get(token, self.unk_id)

    def id_to_token(self, id, **kwargs):
        return self.id2token.get(id, self.unk_token)
