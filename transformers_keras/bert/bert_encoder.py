import numpy as np
import tensorflow as tf

from transformers_keras.bert.bert_attention import BertAttention


def gelu(x):
    """ Gaussian Error Linear Unit.
    Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))
    return x * cdf


def gelu_new(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def swish(x):
    return x * tf.sigmoid(x)


ACT2FN = {
    "gelu": tf.keras.layers.Activation(gelu),
    "relu": tf.keras.activations.relu,
    "swish": tf.keras.layers.Activation(swish),
    "gelu_new": tf.keras.layers.Activation(gelu_new),
}


class BertIntermediate(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.intermediate_size, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
        )
        if isinstance(config.hidden_activation, tf.keras.layers.Activation):
            self.activation = config.hidden_activation
        elif isinstance(config.hidden_activation, str):
            self.activation = ACT2FN[config.hidden_activation]
        else:
            self.activation = ACT2FN['gelu']

    def call(self, inputs, traning=False):
        hidden_states = inputs
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class BertEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(name='BertEncoderLayer')
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name='LayerNorm')

    def call(self, inputs, training=False):
        hidden_states, attention_mask = inputs
        _hidden_states, attention_score = self.attention(inputs=[hidden_states, attention_mask], training=training)
        outputs = self.intermediate(inputs=_hidden_states)
        outputs = self.dense(outputs)
        outputs = self.dropout(outputs, training=training)
        outputs = self.layer_norm(_hidden_states + outputs)
        return outputs, attention_score


class BertEncoder(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(name='BertEncoder')
        self.encoder_layers = [BertEncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def call(self, inputs, training=False):
        hidden_states, attention_mask = inputs
        all_hidden_states = []
        all_attention_scores = []
        for _, encoder in enumerate(self.encoder_layers):
            hidden_states, attention_score = encoder(inputs=[hidden_states, attention_mask], training=training)
            all_hidden_states.append(hidden_states)
            all_attention_scores.append(attention_score)

        return all_hidden_states, all_attention_scores
