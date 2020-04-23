import tensorflow as tf

from .modeling_utils import choose_activation, initialize


class BertConfig(object):

    def __init__(self, **kwargs):
        super().__init__()
        self.vocab_size = kwargs.pop('vocab_size', 21128)  # vocab size of pretrained model `bert-base-chinese`
        self.type_vocab_size = kwargs.pop('type_vocab_size', 2)
        self.hidden_size = kwargs.pop('hidden_size', 768)
        self.num_hidden_layers = kwargs.pop('num_hidden_layers', 12)
        self.num_attention_heads = kwargs.pop('num_attention_heads', 12)
        self.intermediate_size = kwargs.pop('intermediate_size', 3072)
        self.hidden_activation = kwargs.pop('hidden_activation', 'gelu')
        self.hidden_dropout_rate = kwargs.pop('hidden_dropout_rate', 0.1)
        self.attention_dropout_rate = kwargs.pop('attention_dropout_rate', 0.1)
        self.max_position_embeddings = kwargs.pop('max_position_embeddings', 512)
        self.max_sequence_length = kwargs.pop('max_sequence_length', 512)
        self.stddev = kwargs.pop('stddev', 0.02)


class BertEmbedding(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.type_vocab_size = config.type_vocab_size
        self.stddev = config.stddev

        self.position_embedding = tf.keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=initialize(config.stddev),
            name='position_embedding'
        )
        self.token_type_embedding = tf.keras.layers.Embedding(
            config.type_vocab_size,
            config.hidden_size,
            embeddings_initializer=initialize(config.stddev),
            name='token_type_embedding'
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_rate)

    def build(self, input_shape):
        with tf.name_scope('bert_embeddings'):
            self.token_embedding = self.add_weight(
                'weight',
                shape=[self.vocab_size, self.hidden_size],
                initializer=initialize(self.stddev)
            )
        super().build(input_shape)

    def call(self, inputs, mode='embedding', training=None):
        # used for masked lm
        if mode == 'linear':
            return tf.matmul(inputs, self.token_embedding, transpose_b=True)

        input_ids, token_type_ids = inputs
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


class BertAttention(tf.keras.layers.Layer):
    """Multi-head self-attention mechanism from transformer."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        assert self.hidden_size % self.num_attention_heads == 0
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.wq = tf.keras.layers.Dense(
            self.hidden_size, kernel_initializer=initialize(config.stddev), name='query'
        )
        self.wk = tf.keras.layers.Dense(
            self.hidden_size, kernel_initializer=initialize(config.stddev), name='key'
        )
        self.wv = tf.keras.layers.Dense(
            self.hidden_size, kernel_initializer=initialize(config.stddev), name='value'
        )
        self.attention_dropout = tf.keras.layers.Dropout(config.attention_dropout_rate)

        self.dense = tf.keras.layers.Dense(
            self.hidden_size, kernel_initializer=initialize(config.stddev), name='dense'
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name='LayerNorm')
        self.hidden_dropout = tf.keras.layers.Dropout(config.hidden_dropout_rate)

    def call(self, inputs, training=None):
        hidden_states, attention_mask = inputs
        batch_size = tf.shape(hidden_states)[0]

        query = self.wq(hidden_states)
        key = self.wk(hidden_states)
        value = self.wv(hidden_states)

        def _split_heads(x):
            x = tf.reshape(x, shape=[batch_size, -1, self.num_attention_heads, self.attention_head_size])
            return tf.transpose(x, perm=[0, 2, 1, 3])

        query = _split_heads(query)
        key = _split_heads(key)
        value = _split_heads(value)

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


class BertIntermediate(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.intermediate_size, kernel_initializer=initialize(config.stddev)
        )
        self.activation = choose_activation(config.hidden_activation)

    def call(self, inputs):
        hidden_states = inputs
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class BertEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=initialize(config.stddev)
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name='LayerNorm')

    def call(self, inputs, training=False):
        hidden_states, attention_mask = inputs
        _hidden_states, attention_score = self.attention(inputs=[hidden_states, attention_mask])
        outputs = self.intermediate(inputs=_hidden_states)
        outputs = self.dense(outputs)
        outputs = self.dropout(outputs, training=training)
        outputs = self.layer_norm(_hidden_states + outputs)
        return outputs, attention_score


class BertEncoder(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.encoder_layers = [BertEncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def call(self, inputs):
        hidden_states, attention_mask = inputs
        all_hidden_states = []
        all_attention_scores = []
        for _, encoder in enumerate(self.encoder_layers):
            hidden_states, attention_score = encoder(inputs=[hidden_states, attention_mask])
            all_hidden_states.append(hidden_states)
            all_attention_scores.append(attention_score)

        return all_hidden_states, all_attention_scores


class BertPooler(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=initialize(config.stddev),
            activation='tanh',
            name='pooler'
        )

    def call(self, inputs):
        hidden_states = inputs
        # pool the first token: [CLS]
        outputs = self.dense(hidden_states[:, 0])
        return outputs


class BertModel(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.bert_embedding = BertEmbedding(config, **kwargs)
        self.bert_encoder = BertEncoder(config, **kwargs)
        self.bert_pooler = BertPooler(config, **kwargs)

    def call(self, inputs, training=None):
        input_ids, token_type_ids, attention_mask = inputs
        embedding = self.bert_embedding(inputs=[input_ids, token_type_ids], mode='embedding')
        all_hidden_states, all_attention_scores = self.bert_encoder(inputs=[embedding, attention_mask])
        last_hidden_state = all_hidden_states[-1]
        pooled_output = self.bert_pooler(last_hidden_state)
        return last_hidden_state, pooled_output, all_hidden_states, all_attention_scores


class BertMLMHead(tf.keras.layers.Layer):
    """Masked language model for BERT pre-training."""

    def __init__(self, config, embedding, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.embedding = embedding
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=initialize(config.stddev),
            name='dense'
        )
        self.activation = choose_activation(config.hidden_activation)

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12)

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,), initializer='zeros', trainable=True, name='bias')
        super().build(input_shape)

    def call(self, inputs):
        hidden_states = inputs
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.embedding(inputs=hidden_states, mode='linear')
        hidden_states = hidden_states + self.bias
        return hidden_states


class BertNSPHead(tf.keras.layers.Layer):
    """Next sentence prediction for BERT pre-training."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.classifier = tf.keras.layers.Dense(
            2,
            kernel_initializer=initialize(config.stddev),
            name='sequence_relationip'
        )

    def call(self, inputs):
        pooled_output = inputs
        relation = self.classifier(pooled_output)
        return relation


class Bert4PreTraining(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.bert = BertModel(config, name='Bert')
        self.mlm = BertMLMHead(config, self.bert.bert_embedding, name='MLM')
        self.nsp = BertNSPHead(config, name='NSP')

    def call(self, inputs):
        sequence_output, pooled_output, all_hidden_states, all_attention_scores = self.bert(inputs)
        prediction_scores = self.mlm(sequence_output)
        relation_scores = self.nsp(pooled_output)
        return prediction_scores, relation_scores, all_attention_scores[-1]
