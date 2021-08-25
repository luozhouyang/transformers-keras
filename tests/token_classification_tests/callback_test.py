import os
import unittest

from transformers_keras.token_classification.callback import (
    SavedModelForCRFTokenClassification,
    SeqEvalForCRFTokenClassification,
    SeqEvalForTokenClassification,
)
from transformers_keras.token_classification.crf_models import BertCRFForTokenClassification
from transformers_keras.token_classification.models import BertForTokenClassification

BERT_PATH = os.path.join(os.environ["CHINESE_BERT_PATH"], "chinese_roberta_wwm_ext_L-12_H-768_A-12")
VOCAB_PATH = os.path.join(BERT_PATH, "vocab.txt")


class CallbackTest(unittest.TestCase):
    """Callback test"""

    def test_seqeval_for_token_classification(self):
        callback = SeqEvalForTokenClassification.from_conll_files(
            "testdata/conll.txt", "testdata/vocab.bert.txt", "testdata/labels.txt", sep="\\s+"
        )
        model = BertForTokenClassification.from_pretrained(BERT_PATH, override_params={"num_labels": 3})
        callback.model = model
        callback.on_epoch_end(epoch=0, logs=None)

    def test_seqeval_for_crf_token_classification(self):
        callback = SeqEvalForCRFTokenClassification.from_conll_files(
            "testdata/conll.txt", "testdata/vocab.bert.txt", "testdata/labels.txt", sep="\\s+"
        )
        model = BertCRFForTokenClassification.from_pretrained(BERT_PATH, override_params={"num_labels": 3})
        callback.model = model
        callback.on_epoch_end(epoch=0, logs=None)

    def test_savedmodel_for_crf_token_classification(self):
        callback = SavedModelForCRFTokenClassification("models/bert-crf-export")
        model = BertCRFForTokenClassification.from_pretrained(BERT_PATH, override_params={"num_labels": 3})
        callback.model = model
        callback.on_epoch_end(epoch=0, logs=None)


if __name__ == "__main__":
    unittest.main()
