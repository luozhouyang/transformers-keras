import tensorflow as tf

from transformers_keras.modeling_transformer import *


class ModelingTransformerTest(tf.test.TestCase):

    def testTransformerEncoderLayer(self):
        config = TransformerConfig(hidden_size=512, num_attention_heads=8)
        encoder_layer = TransformerEncoderLayer(config)

        inputs = [tf.random.uniform((64, 43, config.hidden_size))] * 3
        output, attn_weights = encoder_layer(inputs=inputs, attn_mask=None)

        self.assertAllEqual(output.shape, [64, 43, config.hidden_size])  # (batch_size, input_seq_len, d_model)
        # (batch_size, num_heads, seq_length, seq_length)
        self.assertAllEqual(attn_weights.shape, [64, config.num_attention_heads, 43, 43])

    def testEncoder(self):
        config = TransformerConfig(num_encoder_layers=2, hidden_size=512, num_attention_heads=8, ffn_size=2048)
        sample_encoder = TransformerEncoder(config)
        inputs = tf.constant([[i for i in range(62)] * 64], shape=(64, 62), dtype=tf.int64)
        output, attn_weights = sample_encoder(inputs=inputs)

        self.assertAllEqual(output.shape, [64, 62, config.hidden_size])  # (batch_size, input_seq_len, d_model)

        # each layers attention weights
        for v in attn_weights:
            self.assertAllEqual(v.shape, [64, config.num_attention_heads, 62, 62])


if __name__ == "__main__":
    tf.test.main()
