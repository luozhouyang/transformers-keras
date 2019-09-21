from transformers_keras.tokenizers.abstract_tokenizer import AbstractTokenizer


class SpaceTokenizer(AbstractTokenizer):
    """Tokenize corpus by SPACE."""

    def __init__(self, config=None):
        super(SpaceTokenizer, self).__init__(config)

    def _process_line(self, line):
        for w in line.split(' '):
            if w in set(self._get_default_config().values()):
                continue
            if w in self._token2id_dict:
                continue
            self._token2id_dict[w] = len(self._token2id_dict.keys())
            self._id2token_dict[len(self._id2token_dict.keys())] = w
