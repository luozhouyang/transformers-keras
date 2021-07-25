import os
import unittest

from transformers_keras.sequence_classify.models import \
    BertForSequenceClassification

CHINESE_BERT_PATH = os.environ['CHINESE_BERT_PATH']

class SequenceClassifyModelsTest(unittest.TestCase):

    def test_bert_for_sequence_classification(self):
        m = BertForSequenceClassification()
        m.summary()
        for w in m.trainable_weights:
            print(w.name)

    def test_bert_for_sequence_classification_from_pretrained(self):
        m = BertForSequenceClassification.from_pretrained(
            os.path.join(CHINESE_BERT_PATH, 'chinese_roberta_wwm_ext_L-12_H-768_A-12'),
            model_params={'num_labels': 2},
        )
        m.summary()
        for w in m.trainable_weights:
            print(w.name)
        for w in m.trainable_weights:
            print(w.numpy())


if __name__ == "__main__":
    unittest.main()
