import json
import logging
import os

import tensorflow as tf

from .layers import MultiHeadAttention
from .modeling_utils import choose_activation, initialize, parse_pretrained_model_files


class BertEmbedding(tf.keras.layers.Layer):

    def __init__(self,
                 vocab_size=1,
                 max_positions=512,
                 hidden_size=768,
                 type_vocab_size=2,
                 dropout_rate=0.2,
                 stddev=0.02,
                 epsilon=1e-12,
                 **kwargs):
        super().__init__(**kwargs)
        assert vocab_size > 0, "vocab_size must greater than 0."
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
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon, name='layer_norm')
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
            self.intermediate_size, kernel_initializer=initialize(self.stddev), name='dense'
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

        self.attention = MultiHeadAttention(
            hidden_size=self.hidden_size, num_attention_heads=self.num_attention_heads, name='mha')
        self.attn_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.attn_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=self.epsilon, name='attn_layer_norm')

        self.intermediate = BertIntermediate(self.intermediate_size, self.activation, self.stddev, name='intermediate')
        self.inter_dense = tf.keras.layers.Dense(
            self.hidden_size, kernel_initializer=initialize(self.stddev), name='dense')
        self.inter_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.inter_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=self.epsilon, name='inter_layer_norm')

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
                name='layer_{}'.format(i)
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
            self.hidden_size, kernel_initializer=initialize(self.stddev), activation='tanh', name='dense')

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


class Bert(tf.keras.layers.Layer):

    def __init__(self,
                 vocab_size=1,
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
        super().__init__(name='main', **kwargs)
        assert vocab_size > 0, "vocab_size must greater than 0."
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
            name='embedding',
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
            name='encoder',
            **kwargs)

        self.bert_pooler = BertPooler(
            hidden_size=self.hidden_size,
            stddev=self.stddev,
            name='pooler',
            **kwargs)

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

    def __init__(self,
                 embedding,
                 vocab_size=-1,
                 hidden_size=768,
                 activation='gelu',
                 epsilon=1e-12,
                 stddev=0.02,
                 **kwargs):
        super().__init__(**kwargs)
        assert vocab_size > 0, "vocab_size must greater than 0."
        self.vocab_size = vocab_size
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.activation = choose_activation(activation)
        self.epsilon = epsilon
        self.stddev = stddev

        self.dense = tf.keras.layers.Dense(self.hidden_size, kernel_initializer=initialize(stddev), name='dense')
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon, name='layer_norm')

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
        self.classifier = tf.keras.layers.Dense(2, kernel_initializer=initialize(self.stddev), name='dense')

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


class BertModel(tf.keras.Model):

    def __init__(self,
                 vocab_size=1,
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
        super(BertModel, self).__init__(name='bert', **kwargs)
        self.max_positions = max_positions
        self.bert = Bert(vocab_size=vocab_size, max_positions=max_positions, hidden_size=hidden_size,
                         type_vocab_size=type_vocab_size, num_layers=num_layers,
                         num_attention_heads=num_attention_heads, intermediate_size=intermediate_size,
                         activation=activation, dropout_rate=dropout_rate, stddev=stddev, epsilon=epsilon, **kwargs)

    def dummy_inputs(self):
        input_ids = tf.constant([0] * self.max_positions, dtype=tf.int64, shape=(1, self.max_positions))
        segment_ids = tf.constant([0] * self.max_positions, dtype=tf.int64, shape=(1, self.max_positions))
        mask = tf.constant([0] * self.max_positions, dtype=tf.int64, shape=(1, self.max_positions))
        return (input_ids, segment_ids, mask)

    @classmethod
    def from_pretrained(cls, pretrained_model_dir, **kwargs):
        config_file, ckpt, _ = parse_pretrained_model_files(pretrained_model_dir)
        model_config = mapping_config(config_file)
        name_mapping = mapping_variables(model_config, load_mlm=False, load_nsp=False)
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
        seqeuence_outputs, pooled_outputs, _, _ = self.bert(inputs)
        return seqeuence_outputs, pooled_outputs


class BertForPretrainingModel(tf.keras.Model):

    def __init__(self,
                 vocab_size=-1,
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
        super(BertForPretrainingModel, self).__init__(name='bert', **kwargs)
        self.max_positions = max_positions
        self.bert = Bert(vocab_size=vocab_size, max_positions=max_positions, hidden_size=hidden_size,
                         type_vocab_size=type_vocab_size, num_layers=num_layers,
                         num_attention_heads=num_attention_heads, intermediate_size=intermediate_size,
                         activation=activation, dropout_rate=dropout_rate, stddev=stddev, epsilon=epsilon, **kwargs)
        self.mlm = BertMLMHead(
            vocab_size=vocab_size,
            embedding=self.bert.bert_embedding,
            hidden_size=hidden_size,
            activation=activation,
            epsilon=epsilon,
            stddev=stddev,
            name='mlm',
            **kwargs)
        self.nsp = BertNSPHead(stddev=stddev, name='nsp', **kwargs)

    def dummy_inputs(self):
        input_ids = tf.constant([0] * self.max_positions, dtype=tf.int64, shape=(1, self.max_positions))
        segment_ids = tf.constant([0] * self.max_positions, dtype=tf.int64, shape=(1, self.max_positions))
        mask = tf.constant([0] * self.max_positions, dtype=tf.int64, shape=(1, self.max_positions))
        return (input_ids, segment_ids, mask)

    @classmethod
    def from_pretrained(cls, pretrained_model_dir, **kwargs):
        config_file, ckpt, _ = parse_pretrained_model_files(pretrained_model_dir)
        model_config = mapping_config(config_file)
        name_mapping = mapping_variables(model_config, load_mlm=True, load_nsp=True)
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
        seqeuence_outputs, pooled_outputs, _, _ = self.bert(inputs)
        predictions = self.mlm(seqeuence_outputs)
        relations = self.nsp(pooled_outputs)
        return predictions, relations


def mapping_config(pretrained_config_file):
    with open(pretrained_config_file, mode='rt', encoding='utf8') as fin:
        config = json.load(fin)

    model_config = {
        'vocab_size': config['vocab_size'],
        'activation': config['hidden_act'],
        'max_positions': config['max_position_embeddings'],
        'hidden_size': config['hidden_size'],
        'type_vocab_size': config['type_vocab_size'],
        'intermediate_size': config['intermediate_size'],
        'dropout_rate': config['hidden_dropout_prob'],
        'stddev': config['initializer_range'],
        'num_layers': config['num_hidden_layers'],
        'num_attention_heads': config['num_attention_heads'],
    }
    return model_config


def mapping_variables(model_config, load_mlm=False, load_nsp=False):
    # model variable name -> pretrained bert variable name
    m = {
        'bert/main/embedding/weight:0': 'bert/embeddings/word_embeddings',
        'bert/main/embedding/position_embedding/embeddings:0': 'bert/embeddings/position_embeddings',
        'bert/main/embedding/token_type_embedding/embeddings:0': 'bert/embeddings/token_type_embeddings',
        'bert/main/embedding/layer_norm/gamma:0': 'bert/embeddings/LayerNorm/gamma',
        'bert/main/embedding/layer_norm/beta:0': 'bert/embeddings/LayerNorm/beta',
    }

    for i in range(model_config['num_layers']):
        # attention
        for n in ['query', 'key', 'value']:
            k = 'bert/main/encoder/layer_{}/mha/{}/kernel:0'.format(i, n)
            v = 'bert/encoder/layer_{}/attention/self/{}/kernel'.format(i, n)
            m[k] = v
            k = 'bert/main/encoder/layer_{}/mha/{}/bias:0'.format(i, n)
            v = 'bert/encoder/layer_{}/attention/self/{}/bias'.format(i, n)
            m[k] = v

        # dense after attention
        for n in ['kernel', 'bias']:
            k = 'bert/main/encoder/layer_{}/mha/dense/{}:0'.format(i, n)
            v = 'bert/encoder/layer_{}/attention/output/dense/{}'.format(i, n)
            m[k] = v
        # layer norm after attention
        for n in ['gamma', 'beta']:
            k = 'bert/main/encoder/layer_{}/attn_layer_norm/{}:0'.format(i, n)
            v = 'bert/encoder/layer_{}/output/LayerNorm/{}'.format(i, n)
            m[k] = v

        # intermediate
        for n in ['kernel', 'bias']:
            k = 'bert/main/encoder/layer_{}/intermediate/dense/{}:0'.format(i, n)
            v = 'bert/encoder/layer_{}/intermediate/dense/{}'.format(i, n)
            m[k] = v

        # output
        for n in ['kernel', 'bias']:
            k = 'bert/main/encoder/layer_{}/dense/{}:0'.format(i, n)
            v = 'bert/encoder/layer_{}/output/dense/{}'.format(i, n)
            m[k] = v

        # layer norm
        for n in ['gamma', 'beta']:
            k = 'bert/main/encoder/layer_{}/inter_layer_norm/{}:0'.format(i, n)
            v = 'bert/encoder/layer_{}/output/LayerNorm/{}'.format(i, n)
            m[k] = v

    # pooler
    for n in ['kernel', 'bias']:
        k = 'bert/main/pooler/dense/{}:0'.format(n)
        v = 'bert/pooler/dense/{}'.format(n)
        m[k] = v

    # masked lm
    if load_mlm:
        m['bert/mlm/bias:0'] = 'cls/predictions/output_bias'
        for n in ['kernel', 'bias']:
            k = 'bert/mlm/dense/{}:0'.format(n)
            v = 'cls/predictions/transform/dense/{}'.format(n)
            m[k] = v
        for n in ['gamma', 'beta']:
            k = 'bert/mlm/layer_norm/{}:0'.format(n)
            v = 'cls/predictions/transform/LayerNorm/{}'.format(n)
            m[k] = v

    # nsp
    if load_nsp:
        m['bert/nsp/dense/kernel:0'] = 'cls/seq_relationship/output_weights'
        m['bert/nsp/dense/bias:0'] = 'cls/seq_relationship/output_bias'

    return m


def zip_weights(model, ckpt, variables_mapping):
    weights, values, names = [], [], []
    for w in model.trainable_weights:
        names.append(w.name)
        weights.append(w)
        v = tf.train.load_variable(ckpt, variables_mapping[w.name])
        if w.name == 'bert/nsp/dense/kernel:0':
            v = v.T
        values.append(v)

    logging.info('weights will be loadded from pretrained checkpoint: \n\t{}'.format('\n\t'.join(names)))

    mapped_values = zip(weights, values)
    return mapped_values
