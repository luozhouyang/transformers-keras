import os
import unittest

from transformers_keras.question_answering.models import \
    BertForQuestionAnsweringModel

CHINESE_BERT_PATH = os.environ['CHINESE_BERT_PATH']


class QuestionAnsweringTest(unittest.TestCase):

    def test_build_qa_model(self):
        m = BertForQuestionAnsweringModel()
        m.summary()
        for w in m.trainable_weights:
            print(w.name)

    def test_load_pretrained_model(self):
        m = BertForQuestionAnsweringModel.from_pretrained(
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
