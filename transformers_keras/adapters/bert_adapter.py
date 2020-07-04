import json
import logging
import os

import tensorflow as tf

from transformers_keras import build_pretraining_bert_model
from transformers_keras.modeling_bert import Bert4PreTraining

from .abstract_adapter import AbstractAdapter


class BertAdapter(AbstractAdapter):

    def __init__(self):
        super().__init__()

    def _parse_files(self, pretrain_model_dir):
        config_file, ckpt, vocab = None, None, None
        if not os.path.exists(pretrain_model_dir):
            logging.info('pretrain model dir: {} is not exists.'.format(pretrain_model_dir))
            return
        for f in os.listdir(pretrain_model_dir):
            if str(f).endswith('config.json'):
                config_file = os.path.join(pretrain_model_dir, f)
            if 'vocab' in str(f):
                vocab = os.path.join(pretrain_model_dir, f)
            if 'ckpt' in str(f):
                n = '.'.join(str(f).split('.')[:-1])
                ckpt = os.path.join(pretrain_model_dir, n)
        return config_file, ckpt, vocab

    def _map_model_config(self, pretrain_config_file):
        with open(pretrain_config_file, mode='rt', encoding='utf8') as fin:
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

    def _variable_loader(self, ckpt):

        def _loader(name):
            return tf.train.load_variable(ckpt, name)
        return _loader

    def _build_variables_mapping(self, num_layers):
        # model variable name -> pretrained bert variable name
        m = {
            'bert/main/embedding/weight:0': 'bert/embeddings/word_embeddings',
            'bert/main/embedding/position_embedding/embeddings:0': 'bert/embeddings/position_embeddings',
            'bert/main/embedding/token_type_embedding/embeddings:0': 'bert/embeddings/token_type_embeddings',
            'bert/main/embedding/layer_normalization/gamma:0': 'bert/embeddings/LayerNorm/gamma',
            'bert/main/embedding/layer_normalization/beta:0': 'bert/embeddings/LayerNorm/beta',
        }

        for i in range(num_layers):
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

    def adapte(self, pretrain_model_dir, **kwargs):
        config, ckpt, vocab = self._parse_files(pretrain_model_dir)

        model_config = self._map_model_config(config)
        model = build_pretraining_bert_model(model_config)

        loader = self._variable_loader(ckpt)

        weights, values, names = [], [], []
        names_mapping = self._build_variables_mapping(model_config['num_layers'])
        for w in model.trainable_weights:
            if w.name not in names_mapping:
                continue
            names.append(w.name)
            weights.append(w)
            v = loader(names_mapping[w.name])
            if w.name == 'bert/nsp/dense/kernel:0':
                v = v.T
            values.append(v)

        logging.info('weights will be loadded from pretrained checkpoint: \n\t{}'.format('\n\t'.join(names)))

        mapped_values = zip(weights, values)
        tf.keras.backend.batch_set_value(mapped_values)

        return model
