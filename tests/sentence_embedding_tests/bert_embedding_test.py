import os
import unittest

from transformers_keras.sentence_embedding.bert_embedding import BertForSentenceEmbedding


class BertEmbeddingTest(unittest.TestCase):
    """Bert for sentence embedding tests."""

    def test_build_bert_for_sentence_embedding(self):
        m = BertForSentenceEmbedding()
        m.summary()
        m.save("models/bert-for-sentence-embedding/1")

    def test_bert_for_sentence_embedding_load_pretrained(self):
        bert_path = os.path.join(os.environ["CHINESE_BERT_PATH"], "chinese_roberta_wwm_ext_L-12_H-768_A-12")
        m = BertForSentenceEmbedding.from_pretrained(bert_path)
        m.summary()
        m.save("models/bert-for-sentence-embedding/2")


if __name__ == "__main__":
    unittest.main()
