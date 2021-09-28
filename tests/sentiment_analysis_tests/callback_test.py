import os
import unittest

from transformers_keras.sentiment_analysis.ate import BertForAspectTermExtraction
from transformers_keras.sentiment_analysis.callback import ExactMatchForAspectTermExtraction, F1ForAspectTermExtraction

BERT_PATH = os.path.join(os.environ["CHINESE_BERT_PATH"], "chinese_roberta_wwm_ext_L-12_H-768_A-12")
VOCAB_PATH = os.path.join(BERT_PATH, "vocab.txt")


class CallbackTest(unittest.TestCase):
    """Callback test"""

    def test_em_for_ate(self):
        callback = ExactMatchForAspectTermExtraction.from_jsonl_files(
            input_files=[os.path.join(os.environ["ZHIJIANG_PATH"], "bert-for-ate.test.jsonl")],
            vocab_file=VOCAB_PATH,
            padding="fixed",
            max_sequence_length=128,
            limit=32,
            context_key="passage",
        )
        model = BertForAspectTermExtraction.from_pretrained(BERT_PATH)
        callback.model = model
        callback.on_epoch_end(epoch=0, logs=None)

    def test_f1_for_ate(self):
        callback = F1ForAspectTermExtraction.from_jsonl_files(
            input_files=[os.path.join(os.environ["ZHIJIANG_PATH"], "bert-for-ate.test.jsonl")],
            vocab_file=VOCAB_PATH,
            padding="fixed",
            max_sequence_length=128,
            limit=32,
            context_key="passage",
        )
        model = BertForAspectTermExtraction.from_pretrained(BERT_PATH)
        callback.model = model
        callback.on_epoch_end(epoch=0, logs=None)


if __name__ == "__main__":
    unittest.main()
