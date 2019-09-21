from transformers_keras.tokenizers.abstract_tokenizer import Tokenizer


class SpaceTokenizer(Tokenizer):
    """Tokenize corpus by SPACE."""

    def __init__(self, config=None):
        super(SpaceTokenizer, self).__init__(config)

        self._id_index = 0

    def _process_line(self, line):
        for w in line.split(' '):
            if w in set(self.default_config.values()):
                continue
            if w in self._tokens2ids_dict:
                continue
            self._tokens2ids_dict[w] = self._id_index
            self._ids2tokens_dict[self._id_index] = w
            self._id_index += 1
