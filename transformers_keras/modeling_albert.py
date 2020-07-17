import json
import logging
import os

import tensorflow as tf

from .modeling_bert import BertEmbedding, BertEncoderLayer
from .modeling_utils import choose_activation, initialize, parse_pretrained_model_files


class AlbertEmbedding(BertEmbedding):

    def __init__(self,
                 vocab_size=-1, max_positions=512, embedding_size=128, type_vocab_size=2, dropout_rate=0.2,
                 stddev=0.02, epsilon=1e-12,
                 **kwargs):
        super().__init__(
            vocab_size=vocab_size,
            max_positions=max_positions, hidden_size=embedding_size, type_vocab_size=type_vocab_size,
            dropout_rate=dropout_rate, stddev=stddev, epsilon=epsilon,
            name='embedding',
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
                name='layer_{}'.format(i)
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
        super(AlbertEncoder, self).__init__(name='encoder', **kwargs)
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
                name='group_{}'.format(i),
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


class Albert(tf.keras.layers.Layer):

    def __init__(self,
                 vocab_size=-1, max_positions=512, embedding_size=128, type_vocab_size=2, num_layers=12, num_groups=1,
                 num_layers_each_group=1, hidden_size=768, num_attention_heads=8, intermediate_size=3072,
                 activation='gelu', dropout_rate=0.2, epsilon=1e-12, stddev=0.02,
                 **kwargs):
        super(Albert, self).__init__(name='main', **kwargs)
        assert vocab_size > 0, "vocab_size must greater than 0."
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
            name='pooler'
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
        base = super(Albert, self).get_config()
        return dict(list(base.items()) + list(config.items()))


class AlbertMLMHead(tf.keras.layers.Layer):

    def __init__(self, embedding, vocab_size=-1, activation='gelu', epsilon=1e-12, stddev=0.02, **kwargs):
        super(AlbertMLMHead, self).__init__(**kwargs)
        assert vocab_size > 0, "vocab_size must greater than 0."
        self.vocab_size = vocab_size
        self.decoder = embedding  # use embedding matrix to decode
        self.activation = choose_activation(activation)
        self.stddev = stddev
        self.epsilon = epsilon
        self.dense = tf.keras.layers.Dense(
            self.decoder.embedding_size, kernel_initializer=initialize(self.stddev), name='dense')
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon, name='layer_norm')

    def build(self, input_shape):
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
        self.classifier = tf.keras.layers.Dense(
            self.num_class, kernel_initializer=initialize(self.stddev), name='dense')

    def call(self, inputs, training=None):
        return self.classifier(inputs)

    def get_config(self):
        config = {
            'num_class': self.num_class,
            'stddev': self.stddev,
        }
        base = super().get_config()
        return dict(list(base.items()) + list(config.items()))


class AlbertModel(tf.keras.Model):

    def __init__(self,
                 vocab_size=-1, max_positions=512, embedding_size=128, type_vocab_size=2, num_layers=12,
                 num_groups=1, num_layers_each_group=1, hidden_size=768, num_attention_heads=8, intermediate_size=3072,
                 activation='gelu', dropout_rate=0.2, epsilon=1e-12, stddev=0.02, **kwargs):
        super(AlbertModel, self).__init__(name='albert', **kwargs)
        self.max_positions = max_positions

        self.albert = Albert(
            vocab_size, max_positions=max_positions, embedding_size=embedding_size,
            type_vocab_size=type_vocab_size, num_layers=num_layers, num_groups=num_groups,
            num_layers_each_group=num_layers_each_group, hidden_size=hidden_size,
            num_attention_heads=num_attention_heads, intermediate_size=intermediate_size,
            activation=activation, dropout_rate=dropout_rate, epsilon=epsilon, stddev=stddev,
            **kwargs)

    def dummy_inputs(self):
        input_ids = tf.constant([0] * self.max_positions, dtype=tf.int64, shape=(1, self.max_positions))
        segment_ids = tf.constant([0] * self.max_positions, dtype=tf.int64, shape=(1, self.max_positions))
        mask = tf.constant([0] * self.max_positions, dtype=tf.int64, shape=(1, self.max_positions))
        return (input_ids, segment_ids, mask)

    @classmethod
    def from_pretrained(cls, pretrained_model_dir, **kwargs):
        config_file, ckpt, _ = parse_pretrained_model_files(pretrained_model_dir)
        model_config = mapping_config(config_file)
        name_mapping = mapping_variables(model_config, load_mlm=False, load_sop=False)
        model = cls(**model_config)
        model(model.dummy_inputs())
        weights_values = zip_weights(model, ckpt, name_mapping)
        tf.keras.backend.batch_set_value(weights_values)
        return model

    def call(self, inputs, training=None):
        if len(inputs) == 1:
            input_ids, token_type_ids, mask = inputs, None, None
        if len(inputs) == 2:
            input_ids, token_type_ids, mask = inputs[0], inputs[1], None
        if len(inputs) >= 3:
            input_ids, token_type_ids, mask = inputs[0], inputs[1], inputs[2]

        if token_type_ids is None:
            token_type_ids = tf.fill(input_ids, 0)
        if mask is None:
            mask = tf.fill(input_ids, 0)

        inputs = (input_ids, token_type_ids, mask)
        seqeuence_outputs, pooled_outputs, _, _ = self.albert(inputs)
        return seqeuence_outputs, pooled_outputs


class AlbertForPretrainingModel(tf.keras.Model):

    def __init__(self,
                 vocab_size=-1, max_positions=512, embedding_size=128, type_vocab_size=2, num_layers=12,
                 num_groups=1, num_layers_each_group=1, hidden_size=768, num_attention_heads=8, intermediate_size=3072,
                 activation='gelu', dropout_rate=0.2, epsilon=1e-12, stddev=0.02, **kwargs):
        super(AlbertForPretrainingModel, self).__init__(name='albert', **kwargs)
        self.max_positions = max_positions

        self.albert = Albert(
            vocab_size, max_positions=max_positions, embedding_size=embedding_size,
            type_vocab_size=type_vocab_size, num_layers=num_layers, num_groups=num_groups,
            num_layers_each_group=num_layers_each_group, hidden_size=hidden_size,
            num_attention_heads=num_attention_heads, intermediate_size=intermediate_size,
            activation=activation, dropout_rate=dropout_rate, epsilon=epsilon, stddev=stddev,
            **kwargs)
        self.mlm = AlbertMLMHead(
            vocab_size=vocab_size, embedding=self.albert.embedding, activation=activation,
            epsilon=epsilon, stddev=stddev,
            name='mlm',
            **kwargs)
        self.sop = AlbertSOPHead(stddev=stddev, name='sop', **kwargs)

    def dummy_inputs(self):
        input_ids = tf.constant([0] * self.max_positions, dtype=tf.int64, shape=(1, self.max_positions))
        segment_ids = tf.constant([0] * self.max_positions, dtype=tf.int64, shape=(1, self.max_positions))
        mask = tf.constant([0] * self.max_positions, dtype=tf.int64, shape=(1, self.max_positions))
        return (input_ids, segment_ids, mask)

    @classmethod
    def from_pretrained(cls, pretrained_model_dir, **kwargs):
        config_file, ckpt, _ = parse_pretrained_model_files(pretrained_model_dir)
        model_config = mapping_config(config_file)
        name_mapping = mapping_variables(model_config, load_mlm=True, load_sop=True)
        model = cls(**model_config)
        model(model.dummy_inputs())
        weights_values = zip_weights(model, ckpt, name_mapping)
        tf.keras.backend.batch_set_value(weights_values)
        return model

    def call(self, inputs, training=None):
        if len(inputs) == 1:
            input_ids, token_type_ids, mask = inputs, None, None
        if len(inputs) == 2:
            input_ids, token_type_ids, mask = inputs[0], inputs[1], None
        if len(inputs) >= 3:
            input_ids, token_type_ids, mask = inputs[0], inputs[1], inputs[2]

        if token_type_ids is None:
            token_type_ids = tf.fill(input_ids, 0)
        if mask is None:
            mask = tf.fill(input_ids, 0)

        inputs = (input_ids, token_type_ids, mask)
        seqeuence_outputs, pooled_outputs, _, _ = self.albert(inputs)
        predictions = self.mlm(seqeuence_outputs)
        relations = self.sop(pooled_outputs)
        return predictions, relations


def mapping_config(pretrained_config_file):
    with open(pretrained_config_file, mode='rt', encoding='utf8') as fin:
        config = json.load(fin)

    model_config = {
        'vocab_size': config['vocab_size'],
        'max_positions': config['max_position_embeddings'],
        'embedding_size': config['embedding_size'],
        'type_vocab_size': config['type_vocab_size'],
        'num_layers': config['num_hidden_layers'],
        'num_groups': config['num_hidden_groups'],
        'num_layers_each_group': config['inner_group_num'],
        'hidden_size': config['hidden_size'],
        'num_attention_heads': config['num_attention_heads'],
        'intermediate_size': config['intermediate_size'],
        'activation': config['hidden_act'],
        'dropout_rate': config['hidden_dropout_prob'],
        'stddev': config['initializer_range'],
    }
    return model_config


def mapping_variables(model_config, load_mlm=False, load_sop=False):
    m = {
        'albert/main/embedding/weight:0': 'bert/embeddings/word_embeddings',
        'albert/main/embedding/position_embedding/embeddings:0': 'bert/embeddings/position_embeddings',
        'albert/main/embedding/token_type_embedding/embeddings:0': 'bert/embeddings/token_type_embeddings',
        'albert/main/embedding/layer_norm/gamma:0': 'bert/embeddings/LayerNorm/gamma',
        'albert/main/embedding/layer_norm/beta:0': 'bert/embeddings/LayerNorm/beta'
    }

    for n in ['kernel', 'bias']:
        k = 'albert/main/encoder/embedding_mapping/{}:0'.format(n)
        v = 'bert/encoder/embedding_hidden_mapping_in/{}'.format(n)
        m[k] = v

    for group in range(model_config['num_groups']):
        for layer in range(model_config['num_layers_each_group']):
            k_prefix = 'albert/main/encoder/group_{}/layer_{}/'.format(group, layer)
            v_prefix = 'bert/encoder/transformer/group_{}/inner_group_{}/'.format(group, layer)

            # attention
            for n in ['query', 'key', 'value']:
                for x in ['kernel', 'bias']:
                    k = k_prefix + 'mha/{}/{}:0'.format(n, x)
                    v = v_prefix + 'attention_1/self/{}/{}'.format(n, x)
                    m[k] = v

            # attention dense
            for n in ['kernel', 'bias']:
                k = k_prefix + 'mha/dense/{}:0'.format(n)
                v = v_prefix + 'attention_1/output/dense/{}'.format(n)
                m[k] = v

            # attention layer norm
            for n in ['gamma', 'beta']:
                k = k_prefix + 'attn_layer_norm/{}:0'.format(n)
                v = v_prefix + 'LayerNorm/{}'.format(n)
                m[k] = v

            # intermediate
            for n in ['kernel', 'bias']:
                k = k_prefix + 'intermediate/dense/{}:0'.format(n)
                v = v_prefix + 'ffn_1/intermediate/dense/{}'.format(n)
                m[k] = v
                k = k_prefix + 'dense/{}:0'.format(n)
                v = v_prefix + 'ffn_1/intermediate/output/dense/{}'.format(n)
                m[k] = v

            # layer norm
            for n in ['gamma', 'beta']:
                k = k_prefix + 'inter_layer_norm/{}:0'.format(n)
                v = v_prefix + 'LayerNorm_1/{}'.format(n)
                m[k] = v

    # pooler
    for n in ['kernel', 'bias']:
        k = 'albert/main/pooler/{}:0'.format(n)
        v = 'bert/pooler/dense/{}'.format(n)
        m[k] = v

    # mlm
    if load_mlm:
        for n in ['kernel', 'bias']:
            k = 'albert/mlm/dense/{}:0'.format(n)
            v = 'cls/predictions/transform/dense/{}'.format(n)
            m[k] = v

        for n in ['gamma', 'beta']:
            k = 'albert/mlm/layer_norm/{}:0'.format(n)
            v = 'cls/predictions/transform/LayerNorm/{}'.format(n)
            m[k] = v

        m['albert/mlm/decoder/bias:0'] = 'cls/predictions/output_bias'

    if load_sop:
        m['albert/sop/dense/kernel:0'] = 'cls/seq_relationship/output_weights'
        m['albert/sop/dense/bias:0'] = 'cls/seq_relationship/output_bias'

    return m


def zip_weights(model, ckpt, variables_mapping):
    weights, values, names = [], [], []
    for w in model.trainable_weights:
        names.append(w.name)
        weights.append(w)
        v = tf.train.load_variable(ckpt, variables_mapping[w.name])
        if w.name == 'albert/sop/dense/kernel:0':
            v = v.T
        values.append(v)

    logging.info('weights will be loadded from pretrained checkpoint: \n\t{}'.format('\n\t'.join(names)))

    mapped_values = zip(weights, values)
    return mapped_values
