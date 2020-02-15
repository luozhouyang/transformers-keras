import tensorflow as tf


class BertAttention(tf.keras.layers.Layer):
    """Multi-head self-attention mechanism from transformer."""

    def __init__(self, config, **kwargs):
        super().__init__(name='BertAttention')
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        assert self.hidden_size % self.num_attention_heads == 0
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.wq = tf.keras.layers.Dense(
            self.hidden_size, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), name='query'
        )
        self.wk = tf.keras.layers.Dense(
            self.hidden_size, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), name='key'
        )
        self.wv = tf.keras.layers.Dense(
            self.hidden_size, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), name='value'
        )
        self.attention_dropout = tf.keras.layers.Dropout(config.attention_dropout_rate)

        self.dense = tf.keras.layers.Dense(
            self.hidden_size, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), name='dense'
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name='LayerNorm')
        self.hidden_dropout = tf.keras.layers.Dropout(config.hidden_dropout_rate)

    def call(self, inputs, training=False):
        hidden_states, attention_mask = inputs
        batch_size = hidden_states.shape.as_list()[0]

        query = self.wq(hidden_states)
        key = self.wk(hidden_states)
        value = self.wv(hidden_states)

        def _split_heads(x):
            x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        query = _split_heads(query)
        key = _split_heads(key)
        value = _split_heads(value)

        attention_score = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(hidden_states.shape[-1], tf.float32)
        attention_score = attention_score / tf.math.sqrt(dk)

        if attention_mask is not None:
            attention_score = attention_score + attention_mask

        attention_score = tf.nn.softmax(attention_score)
        attention_score = self.attention_dropout(attention_score, training=training)

        context = tf.matmul(attention_score, value)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, -1, self.hidden_size))

        # layer norm
        _hidden_states = self.dense(context)
        _hidden_states = self.hidden_dropout(_hidden_states, training=training)
        _hidden_states = self.layer_norm(hidden_states + _hidden_states)

        return _hidden_states, attention_score
