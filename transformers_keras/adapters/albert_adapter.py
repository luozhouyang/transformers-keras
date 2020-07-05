import json
import logging
import os

import tensorflow as tf

from transformers_keras import build_pretraining_albert_model

from .abstract_adapter import AbstractAdapter, AbstractStrategy, PretrainedModelAdapter


def choose_strategy(strategy):
    if isinstance(strategy, AbstractStrategy):
        return strategy
    if 'zh-albert-large' == strategy:
        return ChineseAlbertLargeStrategy()
    else:
        raise ValueError('Invalid strategy: {}'.format(strategy))


class ChineseAlbertLargeStrategy(AbstractStrategy):

    def mapping_config(self, pretrained_config_file):
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

    def build_model(self, model_config):
        return build_pretraining_albert_model(model_config)

    def mapping_variables(self, model_config, model, ckpt):

        model_names = []
        for w in model.trainable_weights:
            model_names.append(w.name)

        logging.info('Model trainable weights: \n\t{}'.format('\n\t'.join(model_names)))
        pretrained_names = [v[0] for v in tf.train.list_variables(ckpt)]
        logging.info('Pretrained  model trainable weights: \n\t{}'.format('\n\t'.join(pretrained_names)))

        logging.info('model weights num: {}, pretrained model weights num: {}'.format(
            len(model.trainable_weights), len(pretrained_names)))

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
        for n in ['kernel', 'bias']:
            k = 'albert/mlm/dense/{}:0'.format(n)
            v = 'cls/predictions/transform/dense/{}'.format(n)
            m[k] = v

        for n in ['gamma', 'beta']:
            k = 'albert/mlm/layer_norm/{}:0'.format(n)
            v = 'cls/predictions/transform/LayerNorm/{}'.format(n)
            m[k] = v

        m['albert/mlm/decoder/bias:0'] = 'cls/predictions/output_bias'
        m['albert/sop/dense/kernel:0'] = 'cls/seq_relationship/output_weights'
        m['albert/sop/dense/bias:0'] = 'cls/seq_relationship/output_bias'

        return m

    def zip_weights(self, model, ckpt, variables_mapping):
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


class AlbertAdapter(PretrainedModelAdapter):

    def __init__(self, strategy='zh-albert-large'):
        self.strategy = choose_strategy(strategy)
        super(AlbertAdapter, self).__init__(strategy=self.strategy)
