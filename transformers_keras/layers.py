import tensorflow as tf


class ScaledDotProductAttention(tf.keras.layers.Layer):

    def __init__(self, attention_dropout_rate=0.1, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(attention_dropout_rate, name='dropout')

    def call(self, query, key, value, attention_mask, training=None):
        query = tf.cast(query, dtype=self.dtype)
        key = tf.cast(key, dtype=self.dtype)
        value = tf.cast(value, dtype=self.dtype)

        score = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(query)[-1], tf.float32)
        score = score / tf.math.sqrt(dk)
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, dtype=self.dtype)
            score += (1.0 - attention_mask) * -10000.0
        attn_weights = tf.nn.softmax(score, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)
        context = tf.matmul(attn_weights, value)
        return context, attn_weights


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, hidden_size=768, num_attention_heads=8, hidden_dropout_rate=0.2, attention_dropout_rate=0.1, epsilon=1e-8, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        assert self.hidden_size % self.num_attention_heads == 0

        self.query_weight = tf.keras.layers.Dense(self.hidden_size, name='query')
        self.key_weight = tf.keras.layers.Dense(self.hidden_size, name='key')
        self.value_weight = tf.keras.layers.Dense(self.hidden_size, name='value')

        self.attention = ScaledDotProductAttention(attention_dropout_rate=attention_dropout_rate, name='self')

        # output block
        self.dense = tf.keras.layers.Dense(self.hidden_size, name='dense')
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=epsilon, name='layer_norm')
        self.dropout = tf.keras.layers.Dropout(hidden_dropout_rate, name='dropout')

    def _split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.hidden_size // self.num_attention_heads))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key, value, attention_mask, training=None):
        origin_input = query  # query == key == value

        batch_size = tf.shape(query)[0]
        query = self._split_heads(self.query_weight(query), batch_size)
        key = self._split_heads(self.key_weight(key), batch_size)
        value = self._split_heads(self.value_weight(value), batch_size)

        context, attn_weights = self.attention(query, key, value, attention_mask)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, [batch_size, -1, self.hidden_size])
        output = self.dense(context)
        output = self.dropout(output, training=training)
        output = self.layer_norm(output + origin_input)
        return output, attn_weights


class PointWiseFeedForwardNetwork(tf.keras.layers.Layer):

    def __init__(self, hidden_size=512, ffn_size=2048, **kwargs):
        super(PointWiseFeedForwardNetwork, self).__init__(**kwargs)
        self.ffn_size = ffn_size
        self.hidden_size = hidden_size
        self.dense1 = tf.keras.layers.Dense(self.ffn_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.hidden_size)

    def call(self, inputs, training=None):
        outputs = self.dense2(self.dense1(inputs))
        return outputs

    def get_config(self):
        config = {
            'ffn_size': self.ffn_size,
            'hidden_size': self.hidden_size,
        }
        p = super(PointWiseFeedForwardNetwork, self).get_config()
        return dict(list(p.items()) + list(config.items()))


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, hidden_size=512, num_attention_heads=8, ffn_size=2048, dropout_rate=0.2, epsilon=1e-6, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.ffn_size = ffn_size
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon

        self.attention = MultiHeadAttention(self.hidden_size, self.num_attention_heads)
        self.attn_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)

        self.ffn = PointWiseFeedForwardNetwork(self.hidden_size, self.ffn_size)
        self.ffn_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.ffn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)

    def call(self, query, key, value, attention_mask, training=None):
        attn, attn_weights = self.attention(query, key, value, attention_mask)
        attn = self.attn_dropout(attn, training=training)
        attn = self.attn_layer_norm(query + attn)

        ffn = self.ffn(attn)
        ffn = self.ffn_dropout(ffn, training=training)
        ffn = self.ffn_layer_norm(ffn + attn)

        return ffn, attn_weights

    def get_config(self):
        config = {
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'ffn_size': self.ffn_size,
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon
        }
        base = super().get_config()
        return dict(list(base.items()) + list(config.items()))


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, hidden_size=512, num_attention_heads=8, ffn_size=2048, dropout_rate=0.2, epsilon=1e-6, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.ffn_size = ffn_size
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon

        self.self_attention = MultiHeadAttention(self.hidden_size, self.num_attention_heads)
        self.self_attn_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)

        self.context_attention = MultiHeadAttention(self.hidden_size, self.num_attention_heads)
        self.context_attn_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.context_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)

        self.ffn = PointWiseFeedForwardNetwork(self.hidden_size, self.ffn_size)
        self.ffn_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.ffn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)

    def call(self, x, enc_outputs, look_ahead_mask, padding_mask, training=None):
        attn1, attn1_weights = self.self_attention(x, x, x, look_ahead_mask)
        attn1 = self.self_attn_dropout(attn1, training=training)
        output1 = self.self_attn_layer_norm(attn1 + x)

        attn2, attn2_weights = self.context_attention(output1, enc_outputs, enc_outputs, padding_mask)
        attn2 = self.context_attn_dropout(attn2, training=training)
        output2 = self.context_attn_layer_norm(attn2 + output1)

        ffn = self.ffn(output2)
        ffn = self.ffn_dropout(ffn, training=training)
        ffn = self.ffn_layer_norm(ffn + attn2)

        return ffn, attn1_weights, attn2_weights

    def get_config(self):
        config = {
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'ffn_size': self.ffn_size,
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon
        }
        base = super(DecoderLayer, self).get_config()
        return dict(list(base.items()) + list(config.items()))
