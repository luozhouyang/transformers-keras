import abc
import logging
import os

import tensorflow as tf


def zip_weights(model, ckpt, variables_mapping, verbose=True):
    weights, values, names = [], [], []
    for w in model.trainable_weights:
        names.append(w.name)
        weights.append(w)
        var = variables_mapping.get(w.name, None)
        if not var:
            logging.warning('model weight: {} does not found in mapping dict.')
            continue
        v = tf.train.load_variable(ckpt, var)
        if w.name == 'bert/nsp/dense/kernel:0':
            v = v.T
        values.append(v)
    if verbose:
        for n in names:
            if n not in variables_mapping:
                logging.warning('model weight: %s not found in ckpt.', n)
                continue
            logging.info('Load model weight: {:70s} <-- {}'.format(n, variables_mapping[n]))

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

    @abc.abstractmethod
    def adapte_config(self, config_file, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def adapte_weights(self, model, config, ckpt, **kwargs):
        raise NotImplementedError()
