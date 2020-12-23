import json
import logging
import os

import tensorflow as tf

from .abstract_adapter import AbstractAdapter, zip_weights


class AlbertAdapter(AbstractAdapter):

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
            'stddev': config['initializer_range'],
        }
        return model_config

    def adapte_weights(self, model, config, ckpt, **kwargs):
        # mapping weight names
        weights_mapping = self._mapping_weight_names(config['num_groups'], config['num_layers_each_group'])
        # zip weights and its' values
        zipped_weights = zip_weights(
            model,
            ckpt,
            weights_mapping,
            verbose=kwargs.get('verbose', True))
        # set values to weights
        tf.keras.backend.batch_set_value(zipped_weights)

    def _mapping_weight_names(self, num_groups, num_layers_each_group):
        mapping = {}

        # embedding
        mapping.update({
            'albert/embeddings/weight:0': 'bert/embeddings/word_embeddings',
            'albert/embeddings/token_type_embeddings/embeddings:0': 'bert/embeddings/token_type_embeddings',
            'albert/embeddings/position_embeddings/embeddings:0': 'bert/embeddings/position_embeddings',
            'albert/embeddings/layer_norm/gamma:0': 'bert/embeddings/LayerNorm/gamma',
            'albert/embeddings/layer_norm/beta:0': 'bert/embeddings/LayerNorm/beta',
            'albert/encoder/embedding_mapping/kernel:0': 'bert/encoder/embedding_hidden_mapping_in/kernel',
            'albert/encoder/embedding_mapping/bias:0': 'bert/encoder/embedding_hidden_mapping_in/bias',
        })

        # encoder
        for group in range(num_groups):
            for layer in range(num_layers_each_group):
                k_prefix = 'albert/encoder/group_{}/layer_{}/'.format(group, layer)
                v_prefix = 'bert/encoder/transformer/group_{}/inner_group_{}/'.format(group, layer)
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

        # pooler
        for n in ['kernel', 'bias']:
            k = 'albert/pooler/{}:0'.format(n)
            v = 'bert/pooler/dense/{}'.format(n)
            mapping[k] = v

        return mapping
