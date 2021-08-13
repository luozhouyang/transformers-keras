import os
import unittest

from transformers_keras.sequence_classification.models import (
    AlbertForSequenceClassification,
    BertForSequenceClassification,
)

CHINESE_BERT_PATH = os.environ["CHINESE_BERT_PATH"]


class SequenceClassificationModelsTest(unittest.TestCase):
    """Sequence classificaton models tests"""

    def test_bert_for_sequence_classification(self):
        m = BertForSequenceClassification()
        m.summary()
        for w in m.trainable_weights:
            print(w.name)

    def test_bert_for_sequence_classification_from_pretrained(self):
        m = BertForSequenceClassification.from_pretrained(
            os.path.join(CHINESE_BERT_PATH, "chinese_roberta_wwm_ext_L-12_H-768_A-12"),
            override_params={"num_labels": 2},
        )
        m.summary()
        for w in m.trainable_weights:
            print(w.name)
        for w in m.trainable_weights:
            print(w.numpy())

    def test_albert_for_sequence_classification(self):
        m = AlbertForSequenceClassification()
        m.summary()
        for w in m.trainable_weights:
            print(w.name)

    def test_albert_for_sequence_classification_from_pretrained(self):
        m = AlbertForSequenceClassification.from_pretrained(
            os.path.join(os.environ["PRETRAINED_MODE_PATH"], "albert_base_zh"),
            override_params={"num_labels": 2},
        )
        m.summary()
        for w in m.trainable_weights:
            print(w.name)
        for w in m.trainable_weights:
            print(w.numpy())


if __name__ == "__main__":
    unittest.main()
