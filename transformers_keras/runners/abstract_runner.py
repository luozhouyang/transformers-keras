import abc
import logging
import os

import tensorflow as tf


class AbstractRunner(abc.ABC):

    def __init__(self, config=None):
        super(AbstractRunner, self).__init__()
        default_config = self._get_default_config()
        if config:
            default_config.update(config)
        self.config = default_config

        self.model = None
        self.optimizer = self._build_optimizer()

        self.ckpt_path = self.config.get('ckpt_path', '/tmp/models')
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        self.ckpt_manager = None

        self.src_vocab_path = os.path.join(self.ckpt_path, 'vocab.src.txt')
        self.tgt_vocab_path = os.path.join(self.ckpt_path, 'vocab.tgt.txt')
        self.src_tokenizer = None
        self.tgt_tokenizer = None

    def train(self):
        raise NotImplementedError()

    def train_and_evaluate(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    def export(self):
        raise NotImplementedError()

    def _build_model(self):
        raise NotImplementedError()

    def _build_optimizer(self):
        raise NotImplementedError()

    def _build_ckpt_manager(self):
        if self.ckpt_manager:
            return self.ckpt_manager
        ckpt = self._build_ckpt()
        # logging.info("Checkpoints will be saved in %s" % self.ckpt_path)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, self.ckpt_path, max_to_keep=self.config.get('max_keep_ckpt', 10))
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            logging.info("Restore ckpt from: %s", ckpt_manager.latest_checkpoint)
        return ckpt_manager

    def _build_ckpt(self):
        raise NotImplementedError()

    def _init_src_tokenizer(self):
        if self.src_tokenizer:
            return
        self.src_tokenizer = self._create_src_tokenizer()
        if os.path.exists(self.src_vocab_path):
            logging.info('Starting to build src tokenizer from vocab...')
            self.src_tokenizer.build_from_vocab(self.src_vocab_path)
            logging.info('Finished to build src tokenizer from vocab.')
            logging.info('src language vocab size: %d\n' % self.src_tokenizer.vocab_size)
        else:
            logging.info('Starting to build src tokenizer from corpus...')
            train_files = self._concat_files(['train_src_files', 'eval_src_files'])
            self.src_tokenizer.build_from_corpus(train_files)
            self.src_tokenizer.save_to_vocab(self.src_vocab_path)
            logging.info('Finished to build src tokenizer from corpus.')
            logging.info('Saved src vocab to: %s' % self.src_vocab_path)
            logging.info('src language vocab size: %d\n' % self.src_tokenizer.vocab_size)

    def _init_tgt_tokenizer(self):
        if self.tgt_tokenizer:
            return
        self.tgt_tokenizer = self._create_tgt_tokenizer()
        if os.path.exists(self.tgt_vocab_path):
            logging.info('Starting to build tgt tokenizer from vocab...')
            self.tgt_tokenizer.build_from_vocab(self.tgt_vocab_path)
            logging.info('Finished to build tgt tokenizer from vocab.')
            logging.info('tgt language vocab size: %d\n' % self.tgt_tokenizer.vocab_size)
        else:
            logging.info('Starting to build tgt tokenizer from corpus...')
            train_files = self._concat_files(['train_tgt_files', 'eval_tgt_files'])
            self.tgt_tokenizer.build_from_corpus(train_files)
            self.tgt_tokenizer.save_to_vocab(self.tgt_vocab_path)
            logging.info('Finished to build tgt tokenizer from corpus.')
            logging.info('Saved tgt vocab to: %s' % self.tgt_vocab_path)
            logging.info('tgt language vocab size: %d\n' % self.tgt_tokenizer.vocab_size)

    def _create_src_tokenizer(self):
        raise NotImplementedError()

    def _create_tgt_tokenizer(self):
        raise NotImplementedError()

    def _concat_files(self, names):
        files = []
        for name in names:
            _files = self.config.get(name, [])
            if not _files:
                continue
            for file in _files:
                files.append(file)
        return files

    @staticmethod
    def _get_default_config():
        c = {

        }
        return c
