from transformers_keras.tokenizers.abstract_tokenizer import AbstractTokenizer


class SpaceTokenizer(AbstractTokenizer):
    """Tokenize corpus by SPACE."""

    def __init__(self, config=None):
        super(SpaceTokenizer, self).__init__(config)
        self._index = self._vocab_size_include_unk

    def _process_line(self, line):
        for w in line.split(' '):
            if w in set(self._get_default_config().values()):
                continue
            if w in self._token2id_dict:
                continue
            self._token2id_dict[w] = self._index
            self._id2token_dict[self._index] = w
            self._index += 1
