import json
import logging
import os

import tensorflow as tf

from transformers_keras import build_pretraining_bert_model
from transformers_keras.modeling_bert import Bert4PreTraining

from .abstract_adapter import AbstractAdapter, AbstractStrategy, PretrainedModelAdapter


def choose_strategy(strategy):
    if isinstance(strategy, AbstractStrategy):
        return strategy
    if 'chinese-bert-base' == strategy:
        return ChineseBertBaseStrategy()
    else:
        raise ValueError('Invalid strategy: {}'.format(strategy))


class ChineseBertBaseStrategy(AbstractStrategy):

    def mapping_config(self, pretrained_config_file):
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

    def build_model(self, model_config):
        model = build_pretraining_bert_model(model_config)
        return model

    def mapping_variables(self, model_config, model, ckpt):
        # model variable name -> pretrained bert variable name
        m = {
            'bert/main/embedding/weight:0': 'bert/embeddings/word_embeddings',
            'bert/main/embedding/position_embedding/embeddings:0': 'bert/embeddings/position_embeddings',
            'bert/main/embedding/token_type_embedding/embeddings:0': 'bert/embeddings/token_type_embeddings',
            'bert/main/embedding/layer_normalization/gamma:0': 'bert/embeddings/LayerNorm/gamma',
            'bert/main/embedding/layer_normalization/beta:0': 'bert/embeddings/LayerNorm/beta',
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
        m['bert/nsp/dense/kernel:0'] = 'cls/seq_relationship/output_weights'
        m['bert/nsp/dense/bias:0'] = 'cls/seq_relationship/output_bias'

        return m

    def zip_weights(self, model, ckpt, variables_mapping):
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


class BertAdapter(PretrainedModelAdapter):

    def __init__(self, strategy='chinese-bert-base'):
        self.strategy = choose_strategy(strategy)
        super(BertAdapter, self).__init__(strategy=self.strategy)
