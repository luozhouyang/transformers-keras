class TokenClassificationLabelTokenizer:
    """Label tokenizer."""

    def __init__(self, label2id, o_token="O", **kwargs) -> None:
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        self.o_token = o_token
        self.o_id = self.label2id[o_token]

    def label_to_id(self, token, **kwargs):
        return self.label2id.get(token, self.o_id)

    def id_to_label(self, id, **kwargs):
        return self.id2label.get(id, self.o_token)

    def labels_to_ids(self, tokens, add_cls=False, add_sep=False, **kwargs):
        ids = [self.label2id.get(token, self.o_id) for token in tokens]
        if add_cls:
            ids = [self.o_id] + ids
        if add_sep:
            ids = ids + [self.o_id]
        return ids

    def ids_to_labels(self, ids, del_cls=True, del_sep=True, **kwargs):
        tokens = [self.id2label.get(_id, self.o_token) for _id in ids]
        if tokens and del_cls and tokens[0] == self.o_token:
            tokens = tokens[1:]
        if tokens and del_sep and tokens[-1] == self.o_token:
            tokens = tokens[:-1]
        return tokens

    @classmethod
    def from_file(cls, vocab_file, o_token="O", **kwargs):
        label2id = cls._read_label_vocab(vocab_file)
        return cls(label2id=label2id, o_token=o_token, **kwargs)

    @classmethod
    def _read_label_vocab(cls, vocab_file):
        m = {}
        with open(vocab_file, mode="rt", encoding="utf-8") as fin:
            for line in fin:
                k = line.strip()
                m[k] = len(m)
        return m
