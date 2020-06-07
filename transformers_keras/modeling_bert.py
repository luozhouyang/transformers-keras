import tensorflow as tf

from .layers import MultiHeadAttention
from .modeling_utils import choose_activation, initialize


class BertEmbedding(tf.keras.layers.Layer):

    def __init__(self,
                 vocab_size,
                 max_positions=512,
                 hidden_size=768,
                 type_vocab_size=2,
                 dropout_rate=0.2,
                 stddev=0.02,
                 epsilon=1e-12,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_positions = max_positions
        self.hidden_size = hidden_size
        self.type_vocab_size = type_vocab_size
        self.dropout_rate = dropout_rate
        self.stddev = stddev
        self.epsilon = epsilon

        self.position_embedding = tf.keras.layers.Embedding(
            self.max_positions,
            self.hidden_size,
            embeddings_initializer=initialize(self.stddev),
            name='position_embedding'
        )
        self.token_type_embedding = tf.keras.layers.Embedding(
            self.type_vocab_size,
            self.hidden_size,
            embeddings_initializer=initialize(self.stddev),
            name='token_type_embedding'
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def build(self, input_shape):
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
        seq_len = input_ids.shape[1]
        position_ids = tf.range(seq_len, dtype=tf.int32)[tf.newaxis, :]
        if token_type_ids is None:
            token_type_ids = tf.fill(input_ids.shape.as_list(), 0)

        position_embeddings = self.position_embedding(position_ids)
        token_type_embeddings = self.token_type_embedding(token_type_ids)
        token_embeddings = tf.gather(self.token_embedding, input_ids)

        embeddings = token_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def get_config(self):
        config = {
            'vocab_size': self.vocab_size,
            'max_positions': self.max_positions,
            'hidden_size': self.hidden_size,
            'type_vocab_size': self.type_vocab_size,
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon,
            'stddev': self.stddev
        }
        base = super().get_config()
        return dict(list(base.items()) + list(config.items()))


class BertIntermediate(tf.keras.layers.Layer):

    def __init__(self, intermediate_size=3072, activation='gelu', stddev=0.02, **kwargs):
        super().__init__(**kwargs)
        self.intermediate_size = intermediate_size
        self.stddev = stddev
        self.dense = tf.keras.layers.Dense(
            self.intermediate_size, kernel_initializer=initialize(self.stddev)
        )
        self.activation = choose_activation(activation)

    def call(self, inputs, training=None):
        hidden_states = inputs
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

    def get_config(self):
        config = {
            'activation': tf.keras.activations.serialize(self.activation),
            'intermediate_size': self.intermediate_size,
            'stddev': self.stddev
        }
        base = super(BertIntermediate, self).get_config()
        return dict(list(base.items()) + list(config.items()))


class BertEncoderLayer(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=8,
                 intermediate_size=3072,
                 activation='gelu',
                 dropout_rate=0.2,
                 epsilon=1e-12,
                 stddev=0.02,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.activation = choose_activation(activation)
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.stddev = stddev

        self.attention = MultiHeadAttention(hidden_size=self.hidden_size, num_attention_heads=self.num_attention_heads)
        self.attn_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)

        self.intermediate = BertIntermediate(self.intermediate_size, self.activation, self.stddev)
        self.inter_dense = tf.keras.layers.Dense(self.hidden_size, kernel_initializer=initialize(self.stddev))
        self.inter_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.inter_layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)

    def call(self, inputs, training=None):
        hidden_states, mask = inputs

        attn_output, attn_weights = self.attention(inputs=(hidden_states, hidden_states, hidden_states, mask))
        attn_output = self.attn_dropout(attn_output, training=training)
        attn_output = self.attn_layer_norm(hidden_states + attn_output)

        outputs = self.intermediate(inputs=attn_output)
        outputs = self.inter_dropout(self.inter_dense(outputs), training=training)
        outputs = self.inter_layer_norm(attn_output + outputs)

        return outputs, attn_weights

    def get_config(self):
        config = {
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'intermediate_size': self.intermediate_size,
            'activation': tf.keras.activations.serialize(self.activation),
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon,
            'stddev': self.stddev
        }
        base = super().get_config()
        return dict(list(base.items()) + list(config.items()))


class BertEncoder(tf.keras.layers.Layer):

    def __init__(self,
                 num_layers=6,
                 hidden_size=768,
                 num_attention_heads=8,
                 intermediate_size=3072,
                 activation='gelu',
                 dropout_rate=0.2,
                 epsilon=1e-12,
                 stddev=0.02,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.activation = choose_activation(activation)
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.stddev = stddev

        self.encoder_layers = [
            BertEncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                activation=activation,
                dropout_rate=dropout_rate,
                epsilon=epsilon,
                stddev=stddev,
                name='BertEncoderLayer{}'.format(i)
            ) for i in range(self.num_layers)
        ]

    def call(self, inputs, training=None):
        hidden_states, attention_mask = inputs
        all_hidden_states = []
        all_attention_scores = []
        for _, encoder in enumerate(self.encoder_layers):
            hidden_states, attention_score = encoder(inputs=(hidden_states, attention_mask))
            all_hidden_states.append(hidden_states)
            all_attention_scores.append(attention_score)

        return hidden_states, all_hidden_states, all_attention_scores

    def get_config(self):
        config = {
            'num_layers': self.num_layers,
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'intermediate_size': self.intermediate_size,
            'activation': tf.keras.activations.serialize(self.activation),
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon,
            'stddev': self.stddev
        }
        base = super().get_config()
        return dict(list(base.items()) + list(config.items()))


class BertPooler(tf.keras.layers.Layer):

    def __init__(self, hidden_size=768, stddev=0.02, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.stddev = stddev
        self.dense = tf.keras.layers.Dense(
            self.hidden_size, kernel_initializer=initialize(self.stddev), activation='tanh')

    def call(self, inputs, training=None):
        hidden_states = inputs
        # pool the first token: [CLS]
        outputs = self.dense(hidden_states[:, 0])
        return outputs

    def get_config(self):
        config = {
            'hidden_size': self.hidden_size,
            'stddev': 0.02,
        }
        base = super().get_config()
        return dict(list(base.items()) + list(config.items()))


class BertModel(tf.keras.layers.Layer):

    def __init__(self,
                 vocab_size,
                 max_positions=512,
                 hidden_size=768,
                 type_vocab_size=2,
                 num_layers=6,
                 num_attention_heads=8,
                 intermediate_size=3072,
                 activation='gelu',
                 dropout_rate=0.2,
                 stddev=0.02,
                 epsilon=1e-12,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_positions = max_positions
        self.hidden_size = hidden_size
        self.type_vocab_size = type_vocab_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.activation = choose_activation(activation)
        self.dropout_rate = dropout_rate
        self.stddev = stddev
        self.epsilon = epsilon

        self.bert_embedding = BertEmbedding(
            vocab_size=self.vocab_size,
            max_positions=self.max_positions,
            hidden_size=self.hidden_size,
            type_vocab_size=self.type_vocab_size,
            dropout_rate=self.dropout_rate,
            stddev=self.stddev,
            epsilon=self.epsilon,
            **kwargs)

        self.bert_encoder = BertEncoder(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            epsilon=self.epsilon,
            stddev=self.stddev,
            **kwargs)

        self.bert_pooler = BertPooler(hidden_size=self.hidden_size, stddev=self.stddev, **kwargs)

    def call(self, inputs, training=None):
        input_ids, token_type_ids, mask = inputs
        mask = mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
        embedding = self.bert_embedding(inputs=(input_ids, token_type_ids), mode='embedding')
        output, all_hidden_states, all_attention_scores = self.bert_encoder(inputs=(embedding, mask))
        pooled_output = self.bert_pooler(output)
        return output, pooled_output, all_hidden_states, all_attention_scores

    def get_config(self):
        config = {
            'num_layers': self.num_layers,
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'intermediate_size': self.intermediate_size,
            'activation': tf.keras.activations.serialize(self.activation),
            'vocab_size': self.vocab_size,
            'max_positions': self.max_positions,
            'type_vocab_size': self.type_vocab_size,
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon,
            'stddev': self.stddev,
        }
        base = super().get_config()
        return dict(list(base.items()) + list(config.items()))


class BertMLMHead(tf.keras.layers.Layer):
    """Masked language model for BERT pre-training."""

    def __init__(self, vocab_size, embedding, hidden_size=768, activation='gelu', epsilon=1e-12, stddev=0.02, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.activation = choose_activation(activation)
        self.epsilon = epsilon
        self.stddev = stddev

        self.dense = tf.keras.layers.Dense(self.hidden_size, kernel_initializer=initialize(stddev))
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,), initializer='zeros', trainable=True, name='bias')
        super().build(input_shape)

    def call(self, inputs, training=None):
        hidden_states = inputs
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.embedding(inputs=hidden_states, mode='linear')
        hidden_states = hidden_states + self.bias
        return hidden_states

    def get_config(self):
        config = {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'activation': tf.keras.activations.serialize(self.activation),
            'epsilon': self.epsilon,
            'stddev': self.stddev
        }
        base = super().get_config()
        return dict(list(base.items()) + list(config.items()))


class BertNSPHead(tf.keras.layers.Layer):
    """Next sentence prediction for BERT pre-training."""

    def __init__(self, stddev=0.02, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev
        self.classifier = tf.keras.layers.Dense(2, kernel_initializer=initialize(self.stddev))

    def call(self, inputs, training=None):
        pooled_output = inputs
        relation = self.classifier(pooled_output)
        return relation

    def get_config(self):
        config = {
            'stddev': self.stddev
        }
        base = super().get_config()
        return dict(list(base.items()) + list(config.items()))


class Bert4PreTraining(tf.keras.layers.Layer):

    def __init__(self,
                 vocab_size,
                 max_positions=512,
                 hidden_size=768,
                 type_vocab_size=2,
                 num_layers=6,
                 num_attention_heads=8,
                 intermediate_size=3072,
                 activation='gelu',
                 dropout_rate=0.2,
                 epsilon=1e-12,
                 stddev=0.02,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_positions = max_positions
        self.hidden_size = hidden_size
        self.type_vocab_size = type_vocab_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.activation = choose_activation(activation)
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.stddev = stddev

        self.bert = BertModel(
            vocab_size=self.vocab_size,
            max_positions=self.max_positions,
            hidden_size=self.hidden_size,
            type_vocab_size=self.type_vocab_size,
            num_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            stddev=self.stddev,
            epsilon=self.epsilon)
        self.mlm = BertMLMHead(
            vocab_size=self.vocab_size,
            embedding=self.bert.bert_embedding,
            hidden_size=self.hidden_size,
            activation=self.activation,
            epsilon=self.epsilon,
            stddev=self.stddev)
        self.nsp = BertNSPHead(stddev=stddev, name='NSP')

    def call(self, inputs, training=None):
        # input_ids, token_type_ids, mask = inputs
        sequence_output, pooled_output, _, all_attention_scores = self.bert(inputs)
        prediction_scores = self.mlm(sequence_output)
        relation_scores = self.nsp(pooled_output)
        return prediction_scores, relation_scores, all_attention_scores

    def get_config(self):
        config = {
            'num_layers': self.num_layers,
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'intermediate_size': self.intermediate_size,
            'activation': tf.keras.activations.serialize(self.activation),
            'vocab_size': self.vocab_size,
            'max_positions': self.max_positions,
            'type_vocab_size': self.type_vocab_size,
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon,
            'stddev': self.stddev,
        }
        base = super().get_config()
        return dict(list(base.items()) + list(config.items()))
