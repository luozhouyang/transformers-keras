import json
import logging
import os

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

    def adapte_weights(self, model, config, ckpt, **kwargs):
        mapping, skipped_weights = {}, []
        ckpt_prefix = kwargs.get('ckpt_prefix', 'bert')
        ckpt_weights = [x[0] for x in tf.train.list_variables(ckpt)]
        self_weights = set([x.name for x in model.trainable_weights])
        for w in ckpt_weights:
            if any(x in w for x in ['embeddings', 'pooler', 'encoder']):
                mw = model.name + w.lstrip(ckpt_prefix) + ':0'
                if mw not in self_weights:
                    logging.warning('weight: %s not in model weights', mw)
                    continue
                mapping[mw] = w

        if self.skip_token_embedding:
            self._skip_weight(mapping, model.name + '/embeddings/word_embeddings:0')
        if self.skip_position_embedding:
            self._skip_weight(mapping, model.name + '/embeddings/position_embeddings:0')
        if self.skip_segment_embedding:
            self._skip_weight(mapping, model.name + '/embeddings/token_type_embeddings:0')
        if self.skip_embedding_layernorm:
            self._skip_weight(mapping, model.name + '/embeddings/LayerNorm/gamma:0')
            self._skip_weight(mapping, model.name + '/embeddings/LayerNorm/beta:0')
        if self.skip_pooler:
            self._skip_weight(mapping, model.name + '/pooler/dense/kernel:0')
            self._skip_weight(mapping, model.name + '/pooler/dense/bias:0')

        # zip weight names and values
        zipped_weights = zip_weights(
            model,
            ckpt,
            mapping,
            **kwargs)
        # set values to weights
        tf.keras.backend.batch_set_value(zipped_weights)

    def _skip_weight(self, mapping, name):
        mapping.pop(name)
        logging.info('Skip load weight: %s', name)
