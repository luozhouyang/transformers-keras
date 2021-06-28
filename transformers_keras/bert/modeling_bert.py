import numpy as np
import tensorflow as tf


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


class BertEmbedding(tf.keras.layers.Layer):

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 type_vocab_size=2,
                 max_positions=512,
                 hidden_dropout_rate=0.1,
                 initializer_range=0.02,
                 epsilon=1e-5,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.embedding_size = hidden_size
        self.initializer_range = initializer_range
        # self.token_embedding = tf.keras.layers.Embedding(
        #     vocab_size,
        #     hidden_size,
        #     embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
        #     name='word_embedding')
        self.segment_embedding = tf.keras.layers.Embedding(
            type_vocab_size,
            hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name='segment_embedding')
        self.position_embedding = tf.keras.layers.Embedding(
            max_positions,
            hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name='position_embedding')

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=epsilon, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(rate=hidden_dropout_rate)

    def build(self, input_shape):
        self.token_embedding = self.add_weight(
            name='word_embedding',
            shape=(self.vocab_size, self.embedding_size),
            dtype=self.dtype,
            initializer=tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)
        )
        return super().build(input_shape)

    @property
    def embedding_table(self):
        return self.token_embedding

    def call(self, inputs, training=None):
        input_ids, token_type_ids, position_ids = None, None, None
        if isinstance(inputs, list) and len(inputs) == 1:
            input_ids = inputs[0]
        if len(inputs) == 2:
            input_ids, token_type_ids = inputs[0], inputs[1]
        if len(inputs) == 3:
            input_ids, token_type_ids, position_ids = inputs[0], inputs[1], inputs[2]
        if token_type_ids is None:
            token_type_ids = tf.zeros_like(input_ids)
        if position_ids is None:
            position_ids = tf.range(0, tf.shape(input_ids)[1], dtype=tf.int32)

        token_embeddings = tf.gather(self.token_embedding, input_ids)
        segment_embeddings = self.segment_embedding(token_type_ids)
        position_embeddings = self.position_embedding(position_ids)

        embeddings = token_embeddings + segment_embeddings + position_embeddings
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def get_config(self):
        base = super().get_config()
        config = {
            'vocab_size': self.vocab_size,
            'type_vocab_size': self.type_vocab_size,
            'embedding_size': self.embedding_size,
            'initializer_range': self.initializer_range
        }
        return dict(list(base.items()) + list(config.items()))


class BertMultiHeadAtttetion(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=8,
                 attention_dropout_rate=0.1,
                 initializer_range=0.02,
                 epsilon=1e-8,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.query_weight = tf.keras.layers.Dense(self.hidden_size, name='query')
        self.key_weight = tf.keras.layers.Dense(self.hidden_size, name='key')
        self.value_weight = tf.keras.layers.Dense(self.hidden_size, name='value')
        self.attention_dropout = tf.keras.layers.Dropout(attention_dropout_rate)

    def _split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.hidden_size // self.num_attention_heads))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _scaled_dot_product_attention(self, query, key, value, attention_mask, training=None):
        query = tf.cast(query, dtype=self.dtype)
        key = tf.cast(key, dtype=self.dtype)
        value = tf.cast(value, dtype=self.dtype)

        score = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(query)[-1], self.dtype)
        score = score / tf.math.sqrt(dk)
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, dtype=self.dtype)
            score += (1.0 - attention_mask) * -10000.0
        attn_weights = tf.nn.softmax(score, axis=-1)
        attn_weights = self.attention_dropout(attn_weights, training=training)
        context = tf.matmul(attn_weights, value)
        return context, attn_weights

    def call(self, query, key, value, attention_mask, training=None):
        batch_size = tf.shape(query)[0]
        query = self._split_heads(self.query_weight(query), batch_size)
        key = self._split_heads(self.key_weight(key), batch_size)
        value = self._split_heads(self.value_weight(value), batch_size)
        attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
        context, attn_weights = self._scaled_dot_product_attention(
            query, key, value, attention_mask, training=training)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, [batch_size, -1, self.hidden_size])
        return context, attn_weights


class BertAttentionOutput(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_size=768,
                 hidden_dropout_rate=0.1,
                 initializer_range=0.02,
                 epsilon=1e-5,
                 **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name='dense')
        self.dropout = tf.keras.layers.Dropout(hidden_dropout_rate, name='dropout')
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=epsilon, name='LayerNorm')

    def call(self, input_states, hidden_states, training=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.layernorm(hidden_states + input_states)
        return hidden_states


class BertAttention(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=8,
                 hidden_dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 initializer_range=0.02,
                 epsilon=1e-5,
                 **kwargs):
        super().__init__(**kwargs)
        self.attention = BertMultiHeadAtttetion(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout_rate=attention_dropout_rate,
            initializer_range=initializer_range,
            epsilon=epsilon,
            name='self')
        self.attention_output = BertAttentionOutput(
            hidden_size=hidden_size,
            hidden_dropout_rate=hidden_dropout_rate,
            initializer_range=initializer_range,
            epsilon=epsilon,
            name='output')

    def call(self, hidden_states, attention_mask, training=None):
        context, attention_weights = self.attention(
            hidden_states, hidden_states, hidden_states, attention_mask, training=training)
        outputs = self.attention_output(hidden_states, context, training=training)
        return outputs, attention_weights


class BertIntermediate(tf.keras.layers.Layer):

    def __init__(self, intermediate_size=3072, activation='gelu', initializer_range=0.02, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            intermediate_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name='dense')
        self.activation = ACT2FN.get(activation, 'relu')

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class BertIntermediateOutput(tf.keras.layers.Layer):

    def __init__(self, hidden_size=768, hidden_dropout_rate=0.1, initializer_range=0.02, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name='dense')
        self.dropout = tf.keras.layers.Dropout(hidden_dropout_rate)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=epsilon, name='LayerNorm')

    def call(self, input_states, hidden_states, training=None):
        hidden_states = self.dropout(self.dense(hidden_states), training=training)
        hidden_states = self.layernorm(hidden_states + input_states)
        return hidden_states


class BertEncoderLayer(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=8,
                 intermediate_size=3072,
                 activation='gelu',
                 hidden_dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 initializer_range=0.02,
                 epsilon=1e-5,
                 **kwargs):
        super().__init__(**kwargs)
        self.attention = BertAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            hidden_dropout_rate=hidden_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            initializer_range=initializer_range,
            epsilon=epsilon,
            name='attention')
        self.intermediate = BertIntermediate(
            intermediate_size=intermediate_size,
            activation=activation,
            initializer_range=initializer_range,
            name='intermediate')
        self.intermediate_output = BertIntermediateOutput(
            hidden_size=hidden_size,
            hidden_dropout_rate=hidden_dropout_rate,
            initializer_range=initializer_range,
            epsilon=epsilon,
            name='output')

    def call(self, hidden_states, attention_mask, training=None):
        attention_outputs, attention_weights = self.attention(
            hidden_states, attention_mask, training=training)
        intermediate_outputs = self.intermediate(attention_outputs)
        outputs = self.intermediate_output(attention_outputs, intermediate_outputs, training=training)
        return outputs, attention_weights


class BertEncoder(tf.keras.layers.Layer):

    def __init__(self,
                 num_layers=12,
                 hidden_size=768,
                 num_attention_heads=8,
                 intermediate_size=3072,
                 activation='gelu',
                 hidden_dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 initializer_range=0.02,
                 epsilon=1e-5,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.layers = [
            BertEncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                activation=activation,
                hidden_dropout_rate=hidden_dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                initializer_range=initializer_range,
                epsilon=epsilon,
                name='layer_{}'.format(i)
            ) for i in range(self.num_layers)
        ]

    def call(self, hidden_states, attention_mask, training=None):
        all_hidden_states, all_attention_weights = [], []
        for i in range(self.num_layers):
            layer = self.layers[i]
            hidden_states, attention_weights = layer(hidden_states, attention_mask, training=training)
            all_hidden_states.append(hidden_states)
            all_attention_weights.append(attention_weights)

        all_hidden_states = tf.stack(all_hidden_states, axis=0)
        all_hidden_states = tf.transpose(all_hidden_states, perm=[1, 0, 2, 3])
        all_attention_weights = tf.stack(all_attention_weights, axis=0)
        all_attention_weights = tf.transpose(all_attention_weights, perm=[1, 0, 2, 3, 4])
        return hidden_states, all_hidden_states, all_attention_weights


class BertPooler(tf.keras.layers.Layer):

    def __init__(self, hidden_size=768, **kwargs):
        super().__init__()
        self.dense = tf.keras.layers.Dense(hidden_size, name='dense')

    def call(self, hidden_states):
        hidden_states = hidden_states[:, 0]
        pooled_output = self.dense(hidden_states)
        pooled_output = tf.nn.tanh(pooled_output)
        return pooled_output


class BertPreTrainedModel(tf.keras.Model):

    def __init__(self, **kwargs):
        self.return_attention_weights = kwargs.pop('return_attention_weights', False)
        self.return_hidden_states = kwargs.pop('return_hidden_states', False)
        super().__init__(**kwargs)

    def dummy_inputs(self):
        input_ids = tf.reshape(tf.range(0, 512, dtype=tf.int32), shape=(1, 512))
        segment_ids = tf.zeros(shape=(1, 512), dtype=tf.int32)
        attention_mask = tf.ones(shape=(1, 512), dtype=tf.int32)
        return input_ids, segment_ids, attention_mask

    def from_pretrained(self, **kwargs):
        pass


class BertModel(BertPreTrainedModel):

    def __init__(self,
                 vocab_size=21128,
                 num_layers=12,
                 num_attention_heads=8,
                 intermediate_size=3072,
                 hidden_size=768,
                 type_vocab_size=2,
                 max_positions=512,
                 hidden_dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 initializer_range=0.02,
                 activation='gelu',
                 epsilon=1e-5,
                 return_hidden_states=False,
                 return_attention_weights=False,
                 **kwargs):
        super().__init__(
            return_hidden_states=return_hidden_states,
            return_attention_weights=return_attention_weights,
            **kwargs)
        self.embedding = BertEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            type_vocab_size=type_vocab_size,
            max_positions=max_positions,
            hidden_dropout_rate=hidden_dropout_rate,
            initializer_range=initializer_range,
            epsilon=epsilon,
            name='embedding')
        self.encoder = BertEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            activation=activation,
            hidden_dropout_rate=hidden_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            initializer_range=initializer_range,
            epsilon=epsilon,
            name='encoder')
        self.pooler = BertPooler(hidden_size=hidden_size, name='pooler')

    def get_embedding_table(self):
        return self.embedding.embedding_table

    def call(self, inputs, training=None):
        input_ids, segment_ids, attention_mask = None, None, None
        if isinstance(inputs, list) and len(inputs) == 1:
            input_ids = inputs[0]
        elif len(inputs) == 2:
            input_ids, segment_ids = inputs[0], inputs[1]
        elif len(inputs) == 3:
            input_ids, segment_ids, attention_mask = inputs[0], inputs[1], inputs[2]
        if segment_ids is None:
            segment_ids = tf.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = tf.cast(tf.greater(input_ids, 0), dtype=tf.int32)

        embedding_outputs = self.embedding([input_ids, segment_ids], training=training)
        sequence_outputs, hidden_states, attention_weights = self.encoder(
            embedding_outputs, attention_mask, training=training)
        pooled_outputs = self.pooler(sequence_outputs)
        outputs = (sequence_outputs, pooled_outputs)
        if self.return_hidden_states:
            outputs += (hidden_states, )
        if self.return_attention_weights:
            outputs += (attention_weights, )
        return outputs


class BertNextSentencePrediction(tf.keras.layers.Layer):

    def __init__(self, num_class=2, **kwargs):
        super().__init__(**kwargs)
        self.classifier = tf.keras.layers.Dense(num_class, name='nsp')

    def call(self, pooled_outputs):
        return self.classifier(pooled_outputs)


class BertMaskedLanguageModel(tf.keras.layers.Layer):

    def __init__(self, embedding_table, **kwargs):
        super().__init__(**kwargs)
        self.embedding_table = embedding_table

    def build(self, input_shape):
        self.vocab_size, hidden_size = self.embedding_table.shape
        self.dense = tf.keras.layers.Dense(
            hidden_size,
            name='transform/dense')
        self.layer_norm = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-12, name='transform/LayerNorm')
        self.bias = self.add_weight(
            'output_bias/bias',
            shape=(self.vocab_size,),
            initializer='zeros',
            trainable=True)

        super().build(input_shape)

    def call(self, inputs):
        sequence_outputs, masked_positions = inputs[0], inputs[1]
        masked_lm_input = self._gather_indexes(sequence_outputs, masked_positions)
        lm_data = self.dense(masked_lm_input)
        lm_data = self.layer_norm(lm_data)
        lm_data = tf.matmul(lm_data, self.embedding_table, transpose_b=True)
        logits = tf.nn.bias_add(lm_data, self.bias)
        masked_positions_length = masked_positions.shape.as_list()[1] or tf.shape(masked_positions)[1]
        logits = tf.reshape(logits, [-1, masked_positions_length, self.vocab_size])
        return logits

    def _gather_indexes(self, sequence_tensor, positions):
        """Gathers the vectors at the specific positions, for performance.
        Args:
            sequence_tensor: Sequence output of shape
            (`batch_size`, `seq_length`, num_hidden) where num_hidden is number of
            hidden units.
            positions: Positions ids of tokens in sequence to mask for pretraining
            of with dimension (batch_size, num_predictions) where
            `num_predictions` is maximum number of tokens to mask out and predict
            per each sequence.
        Returns:
            Masked out sequence tensor of shape (batch_size * num_predictions,
            num_hidden).
        """
        sequence_shape = tf.shape(sequence_tensor)
        batch_size, seq_length = sequence_shape[0], sequence_shape[1]
        width = sequence_tensor.shape.as_list()[2] or sequence_shape[2]

        flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
        flat_positions = tf.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)

        return output_tensor
