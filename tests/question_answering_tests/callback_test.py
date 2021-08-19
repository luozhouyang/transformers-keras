import os
import unittest

from transformers_keras.question_answering.callback import ExactMatchForQuestionAnswering, F1ForQuestionAnswering
from transformers_keras.question_answering.models import BertForQuestionAnswering

BERT_PATH = os.path.join(os.environ["CHINESE_BERT_PATH"], "chinese_roberta_wwm_ext_L-12_H-768_A-12")
VOCAB_PATH = os.path.join(BERT_PATH, "vocab.txt")


class CallbackTest(unittest.TestCase):
    """Callback test"""

    def test_em_for_qa(self):
        callback = ExactMatchForQuestionAnswering.from_jsonl_files(
            input_files=[os.path.join(os.environ["SOGOUQA_PATH"], "sogouqa.test.jsonl")],
            vocab_file=VOCAB_PATH,
            limit=32,
            context_key="passage",
        )
        model = BertForQuestionAnswering.from_pretrained(BERT_PATH)
        callback.model = model
        callback.on_epoch_end(epoch=0, logs=None)

    def test_f1_for_qa(self):
        callback = F1ForQuestionAnswering.from_jsonl_files(
            input_files=[os.path.join(os.environ["SOGOUQA_PATH"], "sogouqa.test.jsonl")],
            vocab_file=VOCAB_PATH,
            limit=32,
            context_key="passage",
        )
        model = BertForQuestionAnswering.from_pretrained(BERT_PATH)
        callback.model = model
        callback.on_epoch_end(epoch=0, logs=None)


if __name__ == "__main__":
    unittest.main()
