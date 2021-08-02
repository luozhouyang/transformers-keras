import os
import unittest

from transformers_keras.question_answering.models import (
    AlbertForQuestionAnswering, BertForQuestionAnswering)

CHINESE_BERT_PATH = os.environ['CHINESE_BERT_PATH']


class QuestionAnsweringTest(unittest.TestCase):

    def test_build_bert_for_qa_model(self):
        m = BertForQuestionAnswering()
        m.summary()
        for w in m.trainable_weights:
            print(w.name)

        m.save('models/bert-for-qa/1')

    def test_load_bert_for_qa_pretrained_model(self):
        m = BertForQuestionAnswering.from_pretrained(
            os.path.join(CHINESE_BERT_PATH, 'chinese_roberta_wwm_ext_L-12_H-768_A-12'),
            override_params={'num_labels': 2},
        )
        m.summary()
        for w in m.trainable_weights:
            print(w.name)
        for w in m.trainable_weights:
            print(w.numpy())

        m.save('models/bert-for-qa/2')

    def test_build_albetr_for_qa_model(self):
        m = AlbertForQuestionAnswering()
        m.summary()
        for w in m.trainable_weights:
            print(w.name)

    def test_load_albert_for_qa_pretrained_model(self):
        m = AlbertForQuestionAnswering.from_pretrained(
            os.path.join(os.environ['PRETRAINED_MODE_PATH'], 'albert_base_zh'),
            override_params={'num_labels': 2},
        )
        m.summary()
        for w in m.trainable_weights:
            print(w.name)
        for w in m.trainable_weights:
            print(w.numpy())


if __name__ == "__main__":
    unittest.main()
