import tensorflow as tf

from .modeling_utils import choose_activation, initialize


def shape(x):
    s = x.shape.as_list()
    d = tf.shape(x)
    return [d[i] if v is None else v for i, v in enumerate(s)]


class AlbertConfig(object):

    def __init__(self, **kwargs):
        super().__init__()
        self.stddev = kwargs.pop('stddev', 0.02)
        self.embedding_size = kwargs.pop('embedding_size', 128)
        self.max_position_embeddings = kwargs.pop('max_position_embeddings', 512)
        self.vocab_size = kwargs.pop('vocab_size', 30000)
        self.type_vocab_size = kwargs.pop('type_vocab_size', 2)
        self.epsilon = kwargs.pop('epsilon', 1e-12)
        self.dropout_rate = kwargs.pop('dropout_rate', 0.2)
        self.hidden_size = kwargs.pop('hidden_size', 4096)
        self.hidden_activation = kwargs.pop('hidden_activation', 'gelu')
        self.num_attention_heads = kwargs.pop('num_attention_heads', 8)
        self.num_hidden_layers = kwargs.pop('num_hidden_layers', 12)
        self.num_hidden_groups = kwargs.pop('num_hidden_groups', 1)
        self.num_layers_each_group = kwargs.pop('num_layers_each_group', 1)
        self.intermediate_size = kwargs.pop('intermediate_size', 3072)


class AlbertEmbedding(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.stddev = config.stddev
        self.max_position_embeddings = config.max_position_embeddings
        self.type_vocab_size = config.type_vocab_size
        self.epsilon = config.epsilon
        self.dropout_rate = config.dropout_rate
        self.position_embeddings = tf.keras.layers.Embedding(
            self.max_position_embeddings,
            self.embedding_size,
            embeddings_initializer=initialize(self.stddev),
            name='position_embeddings'
        )
        self.token_type_embeddings = tf.keras.layers.Embedding(
            self.type_vocab_size,
            self.embedding_size,
            embeddings_initializer=initialize(config.stddev),
            name='type_token_embeddings'
        )
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon, name='layernorm')
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def build(self, input_shape):
        with tf.name_scope('word_embeddings'):
            self.word_embeddings = self.add_weight(
                'weight',
                shape=[self.vocab_size, self.embedding_size],
                initializer=initialize(self.stddev)
            )
        super().build(input_shape)

    def call(self, inputs, mode='embedding', training=None):
        if mode == 'linear':
            batch_size = shape(inputs)[0]
            time_step = shape(inputs)[1]
            x = tf.reshape(inputs, [-1, self.embedding_size])
            logits = tf.matmul(x, self.word_embeddings, transpose_b=True)
            return tf.reshape(logits, [batch_size, time_step, self.vocab_size])

        input_ids, token_type_ids = inputs
        position_ids = tf.range(shape(input_ids)[1], dtype=tf.int32)[tf.newaxis, :]
        if token_type_ids is None:
            token_type_ids = tf.fill(shape(input_ids), 0)

        pos_embedding = self.position_embeddings(position_ids)
        token_type_embedding = self.token_type_embeddings(token_type_ids)
        token_embeddng = tf.gather(self.word_embeddings, input_ids)

        embedding = pos_embedding + token_embeddng + token_type_embedding
        embedding = self.layernorm(embedding)
        embedding = self.dropout(embedding, training=training)
        return embedding

    def get_config(self):
        config = {
            'vocab_size': self.vocab_size,
            'embedding_size': self.embedding_size,
            'stddev': self.stddev,
            'max_position_embeddings': self.max_position_embeddings,
            'type_vocab_size': self.type_vocab_size,
            'epsilon': self.epsilon,
            'dropout_rate': self.dropout_rate
        }
        p = super().get_config()
        return dict(list(p.items()) + list(config.items()))


class AlbertAttention(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        assert self.hidden_size % self.num_attention_heads == 0
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.hidden_size = config.hidden_size
        self.wq = tf.keras.layers.Dense(
            self.hidden_size, kernel_initializer=initialize(config.stddev), name='query'
        )
        self.wk = tf.keras.layers.Dense(
            self.hidden_size, kernel_initializer=initialize(config.stddev), name='key'
        )
        self.wv = tf.keras.layers.Dense(
            self.hidden_size, kernel_initializer=initialize(config.stddev), name='value'
        )
        self.attention_dropout = tf.keras.layers.Dropout(config.dropout_rate)

        self.dense = tf.keras.layers.Dense(
            self.hidden_size, kernel_initializer=initialize(config.stddev), name='dense'
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name='LayerNorm')
        self.hidden_dropout = tf.keras.layers.Dropout(config.dropout_rate)

    def call(self, inputs, training=None):
        hidden_states, attention_mask = inputs
        batch_size = shape(hidden_states)[0]

        def _split_heads(x):
            x = tf.reshape(x, shape=[batch_size, -1, self.num_attention_heads, self.attention_head_size])
            return tf.transpose(x, perm=[0, 2, 1, 3])

        query = _split_heads(self.wq(hidden_states))
        key = _split_heads(self.wk(hidden_states))
        value = _split_heads(self.wv(hidden_states))

        attention_score = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(hidden_states.shape[-1], tf.float32)
        attention_score = attention_score / tf.math.sqrt(dk)

        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32)
            attention_mask = (1.0 - attention_mask) * -10000.0
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

    def get_config(self):
        config = {
            'num_attention_heads': self.num_attention_heads,
            'hidden_size': self.hidden_size,
        }
        p = super().get_config()
        return dict(list(p.items()) + list(config.items()))


class AlbertEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(name='AlbertEncoderLayer', **kwargs)
        self.dropout_rate = config.dropout_rate
        self.attention = AlbertAttention(config, **kwargs)
        self.activation = choose_activation(config.hidden_activation)
        self.ffn = tf.keras.layers.Dense(
            config.intermediate_size, kernel_initializer=initialize(config.stddev), name='ffn'
        )
        self.ffn_output = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=initialize(config.stddev), name='ffn_output'
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.epsilon, name='layer_norm')
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=None):
        hidden_states, attention_mask = inputs
        outputs, attn_weights = self.attention([hidden_states, attention_mask], training=training)
        ffn_output = self.ffn_output(self.activation(self.ffn(outputs)))
        hidden_states = self.dropout(ffn_output, training=training)
        hidden_states = self.layer_norm(outputs + ffn_output)
        return hidden_states, attn_weights

    def get_config(self):
        config = {
            'dropout_rate': self.dropout_rate,
        }
        p = super(AlbertEncoderLayer, self).get_config()
        return dict(list(p.items()) + list(config.items()))


class AlbertEncoderGroup(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(name='AlbertEncoderGroup', **kwargs)
        self.num_layers_each_group = config.num_layers_each_group
        self.encoder_layers = [AlbertEncoderLayer(config, **kwargs) for _ in range(self.num_layers_each_group)]

    def call(self, inputs):
        hidden_states, attn_mask = inputs

        group_hidden_states, group_attn_weights = [], []
        for idx, encoder in enumerate(self.encoder_layers):
            hidden_states, attn_weights = encoder(inputs=[hidden_states, attn_mask])
            group_hidden_states.append(hidden_states)
            group_attn_weights.append(attn_weights)

        return hidden_states, group_hidden_states, group_attn_weights

    def get_config(self):
        config = {
            'num_layers_each_group': self.num_layers_each_group
        }
        p = super(AlbertEncoderGroup, self).get_config()
        return dict(list(p.items()) + list(config.items()))


class AlbertEncoder(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super(AlbertEncoder, self).__init__(**kwargs)
        self.num_hidden_layers = config.num_hidden_layers  # num of encoder layers
        self.num_hidden_groups = config.num_hidden_groups  # num of encoder groups
        self.embedding_mapping = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=initialize(config.stddev),
            name='embedding_mapping'
        )
        self.groups = [
            AlbertEncoderGroup(config, **kwargs) for _ in range(config.num_hidden_groups)
        ]

    def call(self, inputs):
        hidden_states, attention_mask = inputs
        all_hidden_states, all_attn_weights = [], []
        for i in range(self.num_hidden_layers):
            layers_per_group = self.num_hidden_layers // self.num_hidden_groups
            group_index = i // layers_per_group
            hidden_states, group_hidden_states, group_attn_weights = self.groups[group_index](
                inputs=(hidden_states, attention_mask),
            )
            all_hidden_states.extend(group_hidden_states)
            all_attn_weights.extend(group_attn_weights)

        return hidden_states, all_hidden_states, all_attn_weights

    def get_config(self):
        config = {
            'num_hidden_layers': self.num_hidden_layers,
            'num_hidden_groups': self.num_hidden_groups,
        }
        p = super(AlbertEncoder, self).get_config()
        return dict(list(p.items()) + list(config.items()))


class AlbertModel(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super(AlbertModel, self).__init__(**kwargs)
        self.embedding = AlbertEmbedding(config, **kwargs)
        self.encoder = AlbertEncoder(config, **kwargs)
        self.pooler = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=initialize(config.stddev),
            activation='tanh',
            name='pooler'
        )

    def call(self, inputs, training=None):
        input_ids, segment_ids, attention_mask = inputs
        embed = self.embedding(inputs=[input_ids, segment_ids], mode='embedding')
        hidden_states, all_hidden_states, all_attn_weights = self.encoder(inputs=[embed, attention_mask])
        output = self.pooler(hidden_states)
        return output, all_hidden_states, all_attn_weights

    def get_config(self):
        p = super(AlbertModel, self).get_config()
        return dict(list(p.items()))


class AlbertMLMHead(tf.keras.layers.Layer):

    def __init__(self, config, embedding, **kwargs):
        super(AlbertMLMHead, self).__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.decoder = embedding  # use embedding matrix to decode
        self.dense = tf.keras.layers.Dense(
            self.vocab_size,
            kernel_initializer=initialize(config.stddev),
            name='dense'
        )
        self.activation = choose_activation(config.hidden_activation)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.epsilon, name='layer_norm')

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size, ), initializer='zeros', trainable=True, name='bias')
        self.decoder_bias = self.add_weight(
            shape=(self.vocab_size,),
            initializer='zeros',
            trainable=True,
            name='decoder/bias'
        )
        super().build(input_shape)

    def call(self, inputs):
        pooled_output = inputs
        output = self.layer_norm(self.activation(self.dense(pooled_output)))
        output = self.decoder(output, mode='linear') + self.decoder_bias
        output = output + self.bias
        return output

    def get_config(self):
        config = {
            'vocab_size': self.vocab_size
        }
        p = super().get_config()
        return dict(list(p.items()) + list(config.items()))


class AlbertSOPHead(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super(AlbertSOPHead, self).__init__(**kwargs)
        self.num_class = 2
        self.classifier = tf.keras.layers.Dense(
            self.num_class,
            kernel_initializer=config.stddev,
            name='sop'
        )

    def call(self, inputs):
        return self.classifier(inputs)

    def get_config(self):
        config = {
            'num_class': self.num_class
        }
        p = super().get_config()
        return dict(list(p.items()) + list(config.items()))


class Albert4PreTraining(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        self.albert = AlbertModel(config, **kwargs)
        self.mlm = AlbertMLMHead(config, **kwargs)
        self.sop = AlbertSOPHead(config, **kwargs)

    def call(self, inputs):
        outputs, all_hidden_states, all_attn_weights = self.albert(inputs)
        mlm_output = self.mlm(outputs)
        sop_output = self.sop(outputs)
        return mlm_output, sop_output, all_hidden_states, all_attn_weights
