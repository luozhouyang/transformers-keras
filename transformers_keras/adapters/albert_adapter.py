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

    def adapte_weights(self, model, config, ckpt, **kwargs):
        # mapping weight names
        weights_mapping = {}
        ckpt_prefix = kwargs.get('ckpt_prefix', 'bert')
        for cw in [x[0] for x in tf.train.list_variables(ckpt)]:
            if any(x in cw for x in ['embeddings/', 'pooler/']):
                weights_mapping[model.name + cw.lstrip(ckpt_prefix) + ':0'] = cw
                continue
            if 'embedding_hidden_mapping_in/kernel' in cw:
                k = '{}/encoder/embedding_mapping/kernel:0'.format(model.name)
                v = '{}/encoder/embedding_hidden_mapping_in/kernel'.format(ckpt_prefix)
                weights_mapping[k] = v
            if 'embedding_hidden_mapping_in/bias' in cw:
                k = '{}/encoder/embedding_mapping/bias:0'.format(model.name)
                v = '{}/encoder/embedding_hidden_mapping_in/bias'.format(ckpt_prefix)
                weights_mapping[k] = v

        mapping = self._mapping_weight_names(
            config['num_groups'], config['num_layers_each_group'], model_name=model.name)
        weights_mapping.update(mapping)

        # filter
        if self.skip_token_embedding:
            self._skip_weight(weights_mapping, model.name + '/embeddings/word_embeddings:0')
        if self.skip_position_embedding:
            self._skip_weight(weights_mapping, model.name + '/embeddings/position_embeddings:0')
        if self.skip_segment_embedding:
            self._skip_weight(weights_mapping, model.name + '/embeddings/token_type_embeddings:0')
        if self.skip_embedding_layernorm:
            self._skip_weight(weights_mapping, model.name + '/embeddings/LayerNorm/gamma:0')
            self._skip_weight(weights_mapping, model.name + '/embeddings/LayerNorm/beta:0')
        if self.skip_pooler:
            self._skip_weight(weights_mapping, model.name + '/pooler/dense/kernel:0')
            self._skip_weight(weights_mapping, model.name + '/pooler/dense/bias:0')
        if self.skip_embedding_mapping_in:
            self._skip_weight(weights_mapping, model.name + '/encoder/embedding_mapping/kernel:0')
            self._skip_weight(weights_mapping, model.name + '/encoder/embedding_mapping/bias:0')

        # zip weights and its' values
        zipped_weights = zip_weights(
            model,
            ckpt,
            weights_mapping,
            verbose=kwargs.get('verbose', True))
        # set values to weights
        tf.keras.backend.batch_set_value(zipped_weights)

        self_weights = {w.name: w.numpy() for w in model.trainable_weights}
        for k, v in self_weights.items():
            ckpt_key = weights_mapping.get(k, None)
            if not ckpt_key:
                continue
            ckpt_value = tf.train.load_variable(ckpt, ckpt_key)
            if ckpt_value is None:
                logging.warning('ckpt value is None of key: %s', ckpt_key)
            assert np.allclose(v, ckpt_value)

    def _mapping_weight_names(self, num_groups, num_layers_each_group, model_name='albert', ckpt_prefix='bert'):
        logging.info('Using model_name: %s', model_name)
        mapping = {}

        # encoder
        for group in range(num_groups):
            for layer in range(num_layers_each_group):
                k_prefix = '{}/encoder/group_{}/layer_{}/'.format(model_name, group, layer)
                v_prefix = '{}/encoder/transformer/group_{}/inner_group_{}/'.format(ckpt_prefix, group, layer)
                # attention
                for n in ['query', 'key', 'value']:
                    for x in ['kernel', 'bias']:
                        k = k_prefix + 'attention/{}/{}:0'.format(n, x)
                        v = v_prefix + 'attention_1/self/{}/{}'.format(n, x)
                        mapping[k] = v

                # attention dense
                for n in ['kernel', 'bias']:
                    k = k_prefix + 'attention/dense/{}:0'.format(n)
                    v = v_prefix + 'attention_1/output/dense/{}'.format(n)
                    mapping[k] = v

                for n in ['gamma', 'beta']:
                    # attention layer norm
                    k = k_prefix + 'attention/layer_norm/{}:0'.format(n)
                    v = v_prefix + 'LayerNorm/{}'.format(n)
                    mapping[k] = v
                    # albert encoder layer norm
                    k = k_prefix + 'layer_norm/{}:0'.format(n)
                    v = v_prefix + 'LayerNorm_1/{}'.format(n)
                    mapping[k] = v

                for n in ['kernel', 'bias']:
                    # intermediate
                    k = k_prefix + 'ffn/{}:0'.format(n)
                    v = v_prefix + 'ffn_1/intermediate/dense/{}'.format(n)
                    mapping[k] = v
                    # dense
                    k = k_prefix + 'ffn_output/{}:0'.format(n)
                    v = v_prefix + 'ffn_1/intermediate/output/dense/{}'.format(n)
                    mapping[k] = v

        return mapping

    def _skip_weight(self, mapping, name):
        mapping.pop(name)
        logging.info('Skip load weight: %s', name)
