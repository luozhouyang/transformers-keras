import logging
import os
import re
import unittest

from tokenizers import BertWordPieceTokenizer
from transformers_keras.sentence_embedding.callback import ExampleForSpearman, SpearmanForSentenceEmbedding
from transformers_keras.sentence_embedding.simcse_models import HardNegativeSimCSE, SupervisedSimCSE, UnsupervisedSimCSE

BERT_PATH = os.path.join(os.environ["CHINESE_BERT_PATH"], "chinese_roberta_wwm_ext_L-12_H-768_A-12")
VOCAB_PATH = os.path.join(BERT_PATH, "vocab.txt")


class CallbackTest(unittest.TestCase):
    """Callback test"""

    def _read_examples(self):
        examples = []
        p = os.path.join(os.environ["CHINESE_STS_PATH"], "cnsd-sts-test.txt")
        with open(p, mode="rt", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("||")
                if len(parts) != 4:
                    continue
                examples.append(ExampleForSpearman(sentence_a=parts[1], sentence_b=parts[2], label=int(parts[3])))
        logging.info("Load %d examples.", len(examples))
        return examples

    def test_spearman_for_sentence_embedding(self):
        examples = self._read_examples()
        tokenizer = BertWordPieceTokenizer.from_file(VOCAB_PATH, lowercase=True)
        callback = SpearmanForSentenceEmbedding(examples, tokenizer)

        def _compute_spearman(model):
            callback.model = model
            callback.on_epoch_end(epoch=0, logs=None)

        _compute_spearman(HardNegativeSimCSE.from_pretrained(BERT_PATH))
        _compute_spearman(SupervisedSimCSE.from_pretrained(BERT_PATH))
        _compute_spearman(UnsupervisedSimCSE.from_pretrained(BERT_PATH))


if __name__ == "__main__":
    unittest.main()
