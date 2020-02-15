import tensorflow as tf

from transformers_keras.bert.bert_attention import BertAttention
from transformers_keras.bert.bert_config import BertConfig


class BertAttentionTest(tf.test.TestCase):

    def testBertAttention(self):
        config = BertConfig()
        attention = BertAttention(config)

        states = tf.random.uniform(shape=[2, 10, config.hidden_size])
        attention_mask = None
        hidden_states, attention_score = attention(inputs=[states, attention_mask], training=True)

        self.assertAllEqual([2, 10, config.hidden_size], hidden_states.shape)
        self.assertAllEqual([2, config.num_attention_heads, 10, 10], attention_score.shape)


if __name__ == "__main__":
    tf.test.main()
