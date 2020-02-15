
import tensorflow as tf


class BertEmbedding(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(name='BertEmbedding')
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.type_vocab_size = config.type_vocab_size

        self.position_embedding = tf.keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name='position_embedding'
        )
        self.token_type_embedding = tf.keras.layers.Embedding(
            config.type_vocab_size,
            config.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name='token_type_embedding'
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_rate)

    def build(self, input_shape):
        with tf.name_scope('token_embeddings'):
            self.token_embedding = self.add_weight(
                'weight',
                shape=[self.vocab_size, self.hidden_size],
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
            )
        super().build(input_shape)

    def call(self, inputs, training=False, mask=None):
        input_ids, position_ids, token_type_ids = inputs
        if position_ids is None:
            position_ids = tf.range(input_ids.shape[1], dtype=tf.int32)[tf.newaxis, :]
        if token_type_ids is None:
            token_type_ids = tf.fill(input_ids.shape.as_list(), 0)

        position_embeddings = self.position_embedding(position_ids)
        token_type_embeddings = self.token_type_embedding(token_type_ids)
        token_embeddings = tf.gather(self.token_embedding, input_ids)

        embeddings = token_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings
