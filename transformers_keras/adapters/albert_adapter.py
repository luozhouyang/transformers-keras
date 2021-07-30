import json
import logging
import os

import numpy as np
import tensorflow as tf

from .abstract_adapter import AbstractAdapter, zip_weights


class AlbertAdapter(AbstractAdapter):

    def __init__(self, skip_embedding_mapping_in=False, **kwargs):
        super().__init__(**kwargs)
        self.skip_embedding_mapping_in = skip_embedding_mapping_in

    def adapte_config(self, config_file, **kwargs):
        with open(config_file, mode='rt', encoding='utf8') as fin:
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
            'hidden_dropout_rate': config['hidden_dropout_prob'],
            'attention_dropout_rate': config['attention_probs_dropout_prob'],
            'initializer_range': config['initializer_range'],
        }
        return model_config

    def adapte_weights(self, albert, config, ckpt, prefix='', **kwargs):
        ckpt_prefix = kwargs.pop('ckpt_prefix', 'bert')
        model_prefix = prefix + '/' + albert.name if prefix else albert.name
        logging.info('Using model prefix: %s', model_prefix)
        mapping = {}
        self_weight_names = set([x.name for x in albert.trainable_weights])
        ckpt_weight_names = [x[0] for x in tf.train.list_variables(ckpt)]

        mapping.update(self._adapte_embedding_weights(self_weight_names, ckpt_weight_names, model_prefix, ckpt_prefix))
        mapping.update(self._adapte_encoder_weights(
            self_weight_names,
            ckpt_weight_names,
            model_prefix,
            ckpt_prefix,
            num_groups=config['num_groups'],
            num_layers_each_group=config['num_layers_each_group']))

        # skip weights
        self._skip_weights(mapping, model_prefix)

        # zip weight names and values
        zipped_weights = zip_weights(
            albert,
            ckpt,
            mapping,
            **kwargs)
        # set values to weights
        tf.keras.backend.batch_set_value(zipped_weights)

        # check weights
        self._compare_weights(mapping, albert, ckpt, **kwargs)

    def _adapte_embedding_weights(self, self_weight_names, ckpt_weight_names, model_prefix, ckpt_prefix, **kwargs):
        mapping = {}
        for w in ckpt_weight_names:
            # mapping embedding weights
            if any(x in w for x in ['embeddings', 'pooler']):
                mw = model_prefix + w.lstrip(ckpt_prefix) + ':0'
                if mw not in self_weight_names:
                    logging.warning('weight: %s not in model weights', mw)
                    continue
                mapping[mw] = w
            # mapping embedding mapin weights
            if 'embedding_hidden_mapping_in/kernel' in w:
                mw = model_prefix + '/encoder/embedding_mapping/kernel:0'
                if mw not in self_weight_names:
                    logging.warning('weight: %s not in model weights', mw)
                    continue
                mapping[mw] = w
            if 'embedding_hidden_mapping_in/bias' in w:
                mw = model_prefix + '/encoder/embedding_mapping/bias:0'
                if mw not in self_weight_names:
                    logging.warning('weight: %s not in model weights', mw)
                    continue
                mapping[mw] = w
        return mapping

    def _adapte_encoder_weights(self, self_weight_names, ckpt_weight_names, model_prefix, ckpt_prefix, **kwargs):
        mapping = {}
        for group in range(kwargs['num_groups']):
            for layer in range(kwargs['num_layers_each_group']):
                k_prefix = '{}/encoder/group_{}/layer_{}/'.format(model_prefix, group, layer)
                v_prefix = '{}/encoder/transformer/group_{}/inner_group_{}/'.format(ckpt_prefix, group, layer)
                # attention
                for n in ['query', 'key', 'value']:
                    for x in ['kernel', 'bias']:
                        k = k_prefix + 'attention/{}/{}:0'.format(n, x)
                        v = v_prefix + 'attention_1/self/{}/{}'.format(n, x)
                        if k not in self_weight_names:
                            logging.warning('weight: %s not in model weights', k)
                            continue
                        mapping[k] = v

                # attention dense
                for n in ['kernel', 'bias']:
                    k = k_prefix + 'attention/dense/{}:0'.format(n)
                    v = v_prefix + 'attention_1/output/dense/{}'.format(n)
                    if k not in self_weight_names:
                        logging.warning('weight: %s not in model weights', k)
                        continue
                    mapping[k] = v

                for n in ['gamma', 'beta']:
                    # attention layer norm
                    k = k_prefix + 'attention/layer_norm/{}:0'.format(n)
                    v = v_prefix + 'LayerNorm/{}'.format(n)
                    if k not in self_weight_names:
                        logging.warning('weight: %s not in model weights', k)
                        continue
                    mapping[k] = v
                    # albert encoder layer norm
                    k = k_prefix + 'layer_norm/{}:0'.format(n)
                    v = v_prefix + 'LayerNorm_1/{}'.format(n)
                    if k not in self_weight_names:
                        logging.warning('weight: %s not in model weights', k)
                        continue
                    mapping[k] = v

                for n in ['kernel', 'bias']:
                    # intermediate
                    k = k_prefix + 'ffn/{}:0'.format(n)
                    v = v_prefix + 'ffn_1/intermediate/dense/{}'.format(n)
                    if k not in self_weight_names:
                        logging.warning('weight: %s not in model weights', k)
                        continue
                    mapping[k] = v
                    # dense
                    k = k_prefix + 'ffn_output/{}:0'.format(n)
                    v = v_prefix + 'ffn_1/intermediate/output/dense/{}'.format(n)
                    if k not in self_weight_names:
                        logging.warning('weight: %s not in model weights', k)
                        continue
                    mapping[k] = v
        return mapping

    def _skip_weights(self, mapping, model_prefix):
        if self.skip_token_embedding:
            self._skip_weight(mapping, model_prefix + '/embeddings/word_embeddings:0')
        if self.skip_position_embedding:
            self._skip_weight(mapping, model_prefix + '/embeddings/position_embeddings:0')
        if self.skip_segment_embedding:
            self._skip_weight(mapping, model_prefix + '/embeddings/token_type_embeddings:0')
        if self.skip_embedding_layernorm:
            self._skip_weight(mapping, model_prefix + '/embeddings/LayerNorm/gamma:0')
            self._skip_weight(mapping, model_prefix + '/embeddings/LayerNorm/beta:0')
        if self.skip_pooler:
            self._skip_weight(mapping, model_prefix + '/pooler/dense/kernel:0')
            self._skip_weight(mapping, model_prefix + '/pooler/dense/bias:0')

    def _skip_weight(self, mapping, name):
        mapping.pop(name)
        logging.info('Skip load weight: %s', name)

    def _compare_weights(self, mapping, albert, ckpt, **kwargs):
        if not kwargs.get('check_weights', False):
            return

        self_weights = {w.name: w.numpy() for w in albert.trainable_weights}
        for k, v in self_weights.items():
            ckpt_key = mapping.get(k, None)
            if not ckpt_key:
                continue
            ckpt_value = tf.train.load_variable(ckpt, ckpt_key)
            if ckpt_value is None:
                logging.warning('ckpt value is None of key: %s', ckpt_key)
            assert np.allclose(v, ckpt_value)
        logging.info('All weights value are checked.')
