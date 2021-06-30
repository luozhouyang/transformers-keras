import json
import logging
import os

import numpy as np
import tensorflow as tf

from .abstract_adapter import AbstractAdapter, zip_weights


class BertAdapter(AbstractAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def adapte_config(self, config_file, **kwrgs):
        with open(config_file, mode='rt', encoding='utf8') as fin:
            config = json.load(fin)

        model_config = {
            'vocab_size': config['vocab_size'],
            'activation': config['hidden_act'],
            'max_positions': config['max_position_embeddings'],
            'hidden_size': config['hidden_size'],
            'type_vocab_size': config['type_vocab_size'],
            'intermediate_size': config['intermediate_size'],
            'hidden_dropout_rate': config['hidden_dropout_prob'],
            'attention_dropout_rate': config['attention_probs_dropout_prob'],
            'initializer_range': config['initializer_range'],
            'num_layers': config['num_hidden_layers'],
            'num_attention_heads': config['num_attention_heads'],
        }
        return model_config

    def adapte_weights(self, bert, config, ckpt, **kwargs):
        logging.info('Bert model name: %s', bert.name)
        model_prefix = bert.name
        if kwargs.get('model_prefix', ''):
            model_prefix = kwargs['model_prefix'] + '/' + bert.name
        logging.info('Using model prefix: %s', model_prefix)
        mapping = {}
        ckpt_prefix = kwargs.get('ckpt_prefix', 'bert')
        ckpt_weight_names = [x[0] for x in tf.train.list_variables(ckpt)]
        self_weight_names = set([x.name for x in bert.trainable_weights])
        for w in ckpt_weight_names:
            if any(x in w for x in ['embeddings', 'pooler', 'encoder']):
                mw = model_prefix + w.lstrip(ckpt_prefix) + ':0'
                if mw not in self_weight_names:
                    logging.warning('weight: %s not in model weights', mw)
                    continue
                mapping[mw] = w

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

        # zip weight names and values
        zipped_weights = zip_weights(
            bert,
            ckpt,
            mapping,
            **kwargs)
        # set values to weights
        tf.keras.backend.batch_set_value(zipped_weights)

        if not kwargs.get('check_weights', False):
            return

        self_weights = {w.name: w.numpy() for w in bert.trainable_weights}
        for k, v in self_weights.items():
            ckpt_key = mapping.get(k, None)
            if not ckpt_key:
                continue
            ckpt_value = tf.train.load_variable(ckpt, ckpt_key)
            if ckpt_value is None:
                logging.warning('ckpt value is None of key: %s', ckpt_key)
            assert np.allclose(v, ckpt_value)

    def _skip_weight(self, mapping, name):
        mapping.pop(name)
        logging.info('Skip load weight: %s', name)
