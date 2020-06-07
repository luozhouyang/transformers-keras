import tensorflow as tf

from .modeling_bert import BertEmbedding, BertEncoderLayer
from .modeling_utils import choose_activation, initialize


class AlbertEmbedding(BertEmbedding):

    def __init__(self,
                 vocab_size, max_positions=512, embedding_size=128, type_vocab_size=2, dropout_rate=0.2,
                 stddev=0.02, epsilon=1e-12,
                 **kwargs):
        super().__init__(
            vocab_size,
            max_positions=max_positions, hidden_size=embedding_size, type_vocab_size=type_vocab_size,
            dropout_rate=dropout_rate, stddev=stddev, epsilon=epsilon,
            **kwargs)
        # embedding_size is not hidden_size in ALBERT
        self.embedding_size = embedding_size

    def get_config(self):
        config = {
            'embedding_size': self.embedding_size
        }
        base = super().get_config()
        return dict(list(base.items()) + list(config.items()))


class AlbertEncoderLayer(BertEncoderLayer):

    def __init__(self,
                 hidden_size=768, num_attention_heads=8, intermediate_size=3072, activation='gelu',
                 dropout_rate=0.2, epsilon=1e-12, stddev=0.02,
                 **kwargs):
        super().__init__(
            hidden_size=hidden_size, num_attention_heads=num_attention_heads, intermediate_size=intermediate_size,
            activation=activation, dropout_rate=dropout_rate, epsilon=epsilon, stddev=stddev,
            **kwargs)

    def get_config(self):
        return super().get_config()


class AlbertEncoderGroup(tf.keras.layers.Layer):

    def __init__(self,
                 num_layers_each_group=1, hidden_size=768, num_attention_heads=8, intermediate_size=3072,
                 activation='gelu', dropout_rate=0.2, epsilon=1e-12, stddev=0.02,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_layers_each_group = num_layers_each_group
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.activation = choose_activation(activation)
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.stddev = stddev

        self.encoder_layers = [
            AlbertEncoderLayer(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                stddev=self.stddev,
                name='AlbertEncoderLayer{}'.format(i)
            ) for i in range(self.num_layers_each_group)
        ]

    def call(self, inputs, training=None):
        hidden_states, attn_mask = inputs

        group_hidden_states, group_attn_weights = [], []
        for idx, encoder in enumerate(self.encoder_layers):
            hidden_states, attn_weights = encoder(inputs=(hidden_states, attn_mask))
            group_hidden_states.append(hidden_states)
            group_attn_weights.append(attn_weights)

        return hidden_states, group_hidden_states, group_attn_weights

    def get_config(self):
        config = {
            'num_layers_each_group': self.num_layers_each_group,
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'intermediate_size': self.intermediate_size,
            'activation': tf.keras.activations.serialize(self.activation),
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon,
            'stddev': self.stddev,
        }
        base = super(AlbertEncoderGroup, self).get_config()
        return dict(list(base.items()) + list(config.items()))


class AlbertEncoder(tf.keras.layers.Layer):

    def __init__(self,
                 num_layers=12,
                 num_groups=1,
                 num_layers_each_group=1,
                 hidden_size=768,
                 num_attention_heads=8,
                 intermediate_size=3072,
                 activation='gelu',
                 dropout_rate=0.2,
                 epsilon=1e-12,
                 stddev=0.02,
                 **kwargs):
        super(AlbertEncoder, self).__init__(**kwargs)
        self.num_layers = num_layers  # num of encoder layers
        self.num_groups = num_groups  # num of encoder groups
        self.num_layers_each_group = num_layers_each_group
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.activation = choose_activation(activation)
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.stddev = stddev

        self.embedding_mapping = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=initialize(self.stddev),
            name='embedding_mapping'
        )
        self.groups = [
            AlbertEncoderGroup(
                num_layers_each_group=self.num_layers_each_group,
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                stddev=self.stddev,
                name='AlbertEncoderGroup{}'.format(i),
                **kwargs) for i in range(self.num_groups)
        ]

    def call(self, inputs, training=None):
        hidden_states, attention_mask = inputs
        hidden_states = self.embedding_mapping(hidden_states)

        all_hidden_states, all_attn_weights = [], []
        for i in range(self.num_layers):
            layers_per_group = self.num_layers // self.num_groups
            group_index = i // layers_per_group
            hidden_states, group_hidden_states, group_attn_weights = self.groups[group_index](
                inputs=(hidden_states, attention_mask),
            )
            all_hidden_states.extend(group_hidden_states)
            all_attn_weights.extend(group_attn_weights)

        return hidden_states, all_hidden_states, all_attn_weights

    def get_config(self):
        config = {
            'num_layers': self.num_layers,
            'num_groups': self.num_groups,
            'num_layers_each_group': self.num_layers_each_group,
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'intermediate_size': self.intermediate_size,
            'activation': tf.keras.activations.serialize(self.activation),
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon,
            'stddev': self.stddev,
        }
        base = super(AlbertEncoder, self).get_config()
        return dict(list(base.items()) + list(config.items()))


class AlbertModel(tf.keras.layers.Layer):

    def __init__(self,
                 vocab_size, max_positions=512, embedding_size=128, type_vocab_size=2, num_layers=12, num_groups=1,
                 num_layers_each_group=1, hidden_size=768, num_attention_heads=8, intermediate_size=3072,
                 activation='gelu', dropout_rate=0.2, epsilon=1e-12, stddev=0.02,
                 **kwargs):
        super(AlbertModel, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_positions = max_positions
        self.embedding_size = embedding_size
        self.type_vocab_size = type_vocab_size
        self.num_layers = num_layers  # num of encoder layers
        self.num_groups = num_groups  # num of encoder groups
        self.num_layers_each_group = num_layers_each_group
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.activation = choose_activation(activation)
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.stddev = stddev

        self.embedding = AlbertEmbedding(
            vocab_size=self.vocab_size, max_positions=self.max_positions, embedding_size=self.embedding_size,
            type_vocab_size=self.type_vocab_size, dropout_rate=self.dropout_rate, epsilon=self.epsilon,
            stddev=self.stddev,
            **kwargs)

        self.encoder = AlbertEncoder(
            num_layers=self.num_layers, num_groups=self.num_groups, num_layers_each_group=self.num_layers_each_group,
            hidden_size=self.hidden_size, num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size, activation=self.activation, dropout_rate=dropout_rate,
            epsilon=self.epsilon, stddev=self.stddev,
            **kwargs)

        self.pooler = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=initialize(self.stddev),
            activation='tanh',
            name='AlbertPooler'
        )

    def call(self, inputs, training=None):
        input_ids, segment_ids, mask = inputs
        mask = mask[:, tf.newaxis, tf.newaxis, :]
        embed = self.embedding(inputs=(input_ids, segment_ids), mode='embedding')
        outputs, all_hidden_states, all_attn_weights = self.encoder(inputs=(embed, mask))
        # take [CLS]
        pooled_output = self.pooler(outputs[:, 0])
        return outputs, pooled_output, all_hidden_states, all_attn_weights

    def get_config(self):
        config = {
            'vocab_size': self.vocab_size,
            'max_positions': self.max_positions,
            'hidden_size': self.hidden_size,
            'embedding_size': self.embedding_size,
            'type_vocab_size': self.type_vocab_size,
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon,
            'stddev': self.stddev,
            'num_layers': self.num_layers,
            'num_groups': self.num_groups,
            'num_layers_each_group': self.num_layers_each_group,
            'num_attention_heads': self.num_attention_heads,
            'intermediate_size': self.intermediate_size,
            'activation': tf.keras.activations.serialize(self.activation),
        }
        base = super(AlbertModel, self).get_config()
        return dict(list(base.items()) + config.items())


class AlbertMLMHead(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embedding, activation='gelu', epsilon=1e-12, stddev=0.02, **kwargs):
        super(AlbertMLMHead, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.decoder = embedding  # use embedding matrix to decode
        self.activation = choose_activation(activation)
        self.stddev = stddev
        self.epsilon = epsilon
        self.dense = tf.keras.layers.Dense(self.decoder.embedding_size, kernel_initializer=initialize(self.stddev))
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,), initializer='zeros', trainable=True, name='bias')
        self.decoder_bias = self.add_weight(
            shape=(self.vocab_size,),
            initializer='zeros',
            trainable=True,
            name='decoder/bias'
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        pooled_output = inputs
        output = self.layer_norm(self.activation(self.dense(pooled_output)))
        output = self.decoder(output, mode='linear') + self.decoder_bias
        output = output + self.bias
        return output

    def get_config(self):
        config = {
            'vocab_size': self.vocab_size,
            'activation': tf.keras.activations.serialize(self.activation),
            'stddev': self.stddev,
            'epsilon': self.epsilon
        }
        base = super().get_config()
        return dict(list(base.items()) + list(config.items()))


class AlbertSOPHead(tf.keras.layers.Layer):

    def __init__(self, stddev=0.02, **kwargs):
        super(AlbertSOPHead, self).__init__(**kwargs)
        self.num_class = 2
        self.stddev = stddev
        self.classifier = tf.keras.layers.Dense(self.num_class, kernel_initializer=initialize(self.stddev))

    def call(self, inputs, training=None):
        return self.classifier(inputs)

    def get_config(self):
        config = {
            'num_class': self.num_class,
            'stddev': self.stddev,
        }
        base = super().get_config()
        return dict(list(base.items()) + list(config.items()))


class Albert4PreTraining(tf.keras.layers.Layer):

    def __init__(self,
                 vocab_size, max_positions=512, embedding_size=128, type_vocab_size=2, num_layers=12,
                 num_groups=1, num_layers_each_group=1, hidden_size=768, num_attention_heads=8, intermediate_size=3072,
                 activation='gelu', dropout_rate=0.2, epsilon=1e-12, stddev=0.02, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.max_positions = max_positions
        self.embedding_size = embedding_size
        self.type_vocab_size = type_vocab_size
        self.num_layers = num_layers  # num of encoder layers
        self.num_groups = num_groups  # num of encoder groups
        self.num_layers_each_group = num_layers_each_group
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.activation = choose_activation(activation)
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.stddev = stddev

        self.albert = AlbertModel(
            vocab_size, max_positions=self.max_positions, embedding_size=self.embedding_size,
            type_vocab_size=self.type_vocab_size, num_layers=self.num_layers, num_groups=self.num_groups,
            num_layers_each_group=self.num_layers_each_group, hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size,
            activation=self.activation, dropout_rate=self.dropout_rate, epsilon=self.epsilon, stddev=self.stddev,
            **kwargs)
        self.mlm = AlbertMLMHead(
            vocab_size=self.vocab_size, embedding=self.albert.embedding, activation=self.activation,
            epsilon=self.epsilon, stddev=self.stddev,
            **kwargs)
        self.sop = AlbertSOPHead(stddev=self.stddev, **kwargs)

    def call(self, inputs, training=None):
        outputs, pooled_outputs, all_hidden_states, all_attn_weights = self.albert(inputs)
        mlm_output = self.mlm(outputs)
        sop_output = self.sop(pooled_outputs)
        return mlm_output, sop_output, all_hidden_states, all_attn_weights

    def get_config(self):
        config = {
            'vocab_size': self.vocab_size,
            'max_positions': self.max_positions,
            'hidden_size': self.hidden_size,
            'embedding_size': self.embedding_size,
            'type_vocab_size': self.type_vocab_size,
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon,
            'stddev': self.stddev,
            'num_layers': self.num_layers,
            'num_groups': self.num_groups,
            'num_layers_each_group': self.num_layers_each_group,
            'num_attention_heads': self.num_attention_heads,
            'intermediate_size': self.intermediate_size,
            'activation': tf.keras.activations.serialize(self.activation),
        }
        base = super().get_config()
        return dict(list(base.items()) + list(config.items()))
