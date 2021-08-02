import os
import unittest

from transformers_keras.token_classification.models import AlbertForTokenClassification, BertForTokenClassification


class ModelsTest(unittest.TestCase):
    """Model tests"""

    def test_bert_for_token_classification(self):
        m = BertForTokenClassification(4)
        m.summary()
        m.save("models/bert-for-token-classification/1")

        bert_path = os.path.join(os.environ["CHINESE_BERT_PATH"], "chinese_roberta_wwm_ext_L-12_H-768_A-12")
        m = BertForTokenClassification.from_pretrained(bert_path, override_params={"num_labels": 4})
        m.summary()
        m.save("models/bert-for-token-classification/2")

    def test_albert_for_token_classification(self):
        m = AlbertForTokenClassification(4)
        m.summary()
        m.save("models/albert-for-token-classification/1")

        albert_path = os.path.join(os.environ["GOOGLE_ALBERT_PATH"], "albert-base-zh")
        m = AlbertForTokenClassification.from_pretrained(albert_path, override_params={"num_labels": 4})
        m.summary()
        m.save("models/albert-for-token-classification/2")


if __name__ == "__main__":
    unittest.main()
