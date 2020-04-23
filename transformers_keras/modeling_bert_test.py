import tensorflow as tf

from .modeling_bert import *


class ModelingBertTest(tf.test.TestCase):

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

    def testBertAttention(self):
        config = BertConfig()
        attention = BertAttention(config)

        states = tf.random.uniform(shape=[2, 10, config.hidden_size])
        attention_mask = None
        hidden_states, attention_score = attention(inputs=[states, attention_mask], training=True)

        self.assertAllEqual([2, 10, config.hidden_size], hidden_states.shape)
        self.assertAllEqual([2, config.num_attention_heads, 10, 10], attention_score.shape)

    def testBertEncoder(self):
        config = BertConfig()
        encoder = BertEncoder(config)

        hidden_states = tf.random.uniform((2, 10, config.hidden_size))
        attnion_mask = None
        all_hidden_states, all_attention_scores = encoder(inputs=[hidden_states, attnion_mask], training=True)

        self.assertAllEqual(config.num_hidden_layers, len(all_hidden_states))
        for state in all_hidden_states:
            self.assertAllEqual([2, 10, config.hidden_size], state.shape)

        self.assertAllEqual(config.num_hidden_layers, len(all_attention_scores))
        for attention in all_attention_scores:
            self.assertAllEqual([2, config.num_attention_heads, 10, 10], attention.shape)


if __name__ == "__main__":
    tf.test.main()
