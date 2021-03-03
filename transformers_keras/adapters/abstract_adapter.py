import abc
import logging
import os

import tensorflow as tf


def zip_weights(model, ckpt, variables_mapping, verbose=True):
    weights, values = [], []
    for w in model.trainable_weights:
        var = variables_mapping.get(w.name, None)
        if not var:
            logging.warning(
                'Skip weight: {}, which not found in mapping, thus will not load weight from ckpt.'.format(w.name))
            continue
        v = tf.train.load_variable(ckpt, var)
        if w.name == 'bert/nsp/dense/kernel:0':
            v = v.T
        weights.append(w)
        values.append(v)
        if verbose:
            logging.info('Load weight: {:50s} <-- {}'.format(w.name, variables_mapping[w.name]))

    mapped_values = zip(weights, values)
    return mapped_values


def parse_pretrained_model_files(pretrained_model_dir):
    config_file, ckpt, vocab = None, None, None
    pretrained_model_dir = os.path.abspath(pretrained_model_dir)
    if not os.path.exists(pretrained_model_dir):
        logging.info('pretrain model dir: {} is not exists.'.format(pretrained_model_dir))
        return config_file, ckpt, vocab
    for f in os.listdir(pretrained_model_dir):
        if str(f).endswith('config.json'):
            config_file = os.path.join(pretrained_model_dir, f)
        if 'vocab' in str(f):
            vocab = os.path.join(pretrained_model_dir, f)
        if 'ckpt' in str(f):
            n = '.'.join(str(f).split('.')[:-1])
            ckpt = os.path.join(pretrained_model_dir, n)
    return config_file, ckpt, vocab


class AbstractAdapter(abc.ABC):

    def __init__(self,
                 skip_token_embedding=False,
                 skip_position_embedding=False,
                 skip_segment_embedding=False,
                 skip_embedding_layernorm=False,
                 skip_pooler=False,
                 **kwargs):
        super().__init__()
        self.skip_token_embedding = skip_token_embedding
        self.skip_position_embedding = skip_position_embedding
        self.skip_segment_embedding = skip_segment_embedding
        self.skip_embedding_layernorm = skip_embedding_layernorm
        self.skip_pooler = skip_pooler

    @abc.abstractmethod
    def adapte_config(self, config_file, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def adapte_weights(self, model, config, ckpt, **kwargs):
        raise NotImplementedError()
