import tensorflow as tf

from transformers_keras.bert.bert_config import BertConfig
from transformers_keras.bert.bert_encoder import BertEncoder


class BertEncoderTest(tf.test.TestCase):

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
