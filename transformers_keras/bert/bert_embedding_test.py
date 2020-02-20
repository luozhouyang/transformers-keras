import tensorflow as tf

from transformers_keras.bert.bert_config import BertConfig
from transformers_keras.bert.bert_embedding import BertEmbedding


class BertEmbeddingTest(tf.test.TestCase):

    def testBertEmbeddings(self):
        config = BertConfig()
        embedding_layer = BertEmbedding(config)
        inputs = [
            tf.constant([[0, 2, 3, 4, 5, 1]]),
            # tf.constant([[0, 1, 2, 3, 4, 5]]),
            tf.constant([[0, 0, 0, 1, 1, 1]]),
        ]
        embeddings = embedding_layer(inputs, training=True)

        self.assertAllEqual([1, 6, config.hidden_size], embeddings.shape)


if __name__ == "__main__":
    tf.test.main()
