import tensorflow as tf
from transformers_keras.modeling_bert import BertModel, BertPretrainedModel
from transformers_keras.modeling_utils import choose_activation, shape_list


class BertPredictionHeadTransform(tf.keras.layers.Layer):
    """Transform layer for prediction head."""

    def __init__(self, hidden_size=768, activation="gelu", initializer_range=0.02, epsilon=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="dense",
        )
        self.activation = choose_activation(activation)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=epsilon, name="LayerNorm")

    def call(self, hidden_states, training=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        return hidden_states


class BertPredictionHead(tf.keras.layers.Layer):
    """Prediction head"""

    def __init__(
        self,
        embedding_table,
        vocab_size=21128,
        hidden_size=768,
        activation="gelu",
        initializer_range=0.02,
        epsilon=1e-12,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding_table = embedding_table
        self.transform = BertPredictionHeadTransform(
            hidden_size=hidden_size,
            activation=activation,
            initializer_range=initializer_range,
            epsilon=epsilon,
            name="transform",
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,), initializer="zeros", trainable=True, name="output_bias")
        super().build(input_shape)

    def call(self, hidden_states, training=None):
        hidden_states = self.transform(hidden_states)
        seq_length = shape_list(hidden_states)[1]
        hidden_states = tf.reshape(hidden_states, shape=[-1, self.hidden_size])
        hidden_states = tf.matmul(hidden_states, self.embedding_table, transpose_b=True)
        hidden_states = tf.reshape(hidden_states, shape=[-1, seq_length, self.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)
        return hidden_states


class BertMaskedLanguageModelHead(tf.keras.layers.Layer):
    """Masked language model head for BERT."""

    def __init__(
        self,
        embedding_table,
        vocab_size=21128,
        hidden_size=768,
        activation="gelu",
        initializer_range=0.02,
        epsilon=1e-12,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.prediction = BertPredictionHead(
            embedding_table=embedding_table,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            activation=activation,
            initializer_range=initializer_range,
            epsilon=epsilon,
            name="predictions",
        )

    def call(self, sequence_output, training=None):
        outputs = self.prediction(sequence_output)
        return outputs


class BertForMaskedLanguageModel(BertPretrainedModel):
    """Bert for masked language model."""

    def __init__(
        self,
        vocab_size=21128,
        max_positions=512,
        hidden_size=768,
        type_vocab_size=2,
        num_layers=6,
        num_attention_heads=8,
        intermediate_size=3072,
        activation="gelu",
        hidden_dropout_rate=0.2,
        attention_dropout_rate=0.1,
        initializer_range=0.02,
        epsilon=1e-12,
        **kwargs
    ):
        input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
        segment_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="segment_ids")
        attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
        bert_model = BertModel(
            vocab_size=vocab_size,
            max_positions=max_positions,
            hidden_size=hidden_size,
            type_vocab_size=type_vocab_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            activation=activation,
            hidden_dropout_rate=hidden_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            initializer_range=initializer_range,
            epsilon=epsilon,
            name="bert",
        )
        sequence_output, _, _, _ = bert_model(input_ids, segment_ids, attention_mask)
        embedding_table = bert_model.get_embedding_table()
        mlm = BertMaskedLanguageModelHead(
            embedding_table,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            activation=activation,
            initializer_range=initializer_range,
            epsilon=epsilon,
            name="cls",
        )
        pred = mlm(sequence_output)
        pred = tf.keras.layers.Lambda(lambda x: x, name="predictions")(pred)
        super().__init__(inputs=[input_ids, segment_ids, attention_mask], outputs=[pred], **kwargs)

        self.epsilon = epsilon

    def train_step(self, data):
        x, y = data
        input_ids, segment_ids, attention_mask = x["input_ids"], x["segment_ids"], x["attention_mask"]
        # masked_ids is golden label, masked_pos is position masking matrix
        masked_pos, y_true = y["masked_pos"], y["masked_ids"]
        with tf.GradientTape() as tape:
            y_pred = self(inputs=[input_ids, segment_ids, attention_mask], training=True)
            loss = self._compute_loss(y_true, y_pred, masked_pos)
        # Compute gradients
        trainable_vars = self.trainable_variables
        # print_variables(trainable_vars)
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y_true, y_pred)
        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

    def _compute_loss(self, y_true, y_pred, masked_pos):
        # masked_pos shape: (batch_size, seq_len)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        masked_pos = tf.cast(masked_pos, dtype=loss.dtype)
        loss *= masked_pos
        return tf.reduce_sum(loss) / (tf.reduce_sum(masked_pos) + self.epsilon)

    def test_step(self, data):
        x, y = data
        input_ids, segment_ids, attention_mask = x["input_ids"], x["segment_ids"], x["attention_mask"]
        masked_pos, y_true = y["masked_positions"], y["masked_ids"]
        y_pred = self(inputs=[input_ids, segment_ids, attention_mask], training=False)
        loss = self._compute_loss(y_true, y_pred, masked_pos)
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y_true, y_pred)
        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results
