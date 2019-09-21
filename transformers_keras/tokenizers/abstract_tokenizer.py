import abc
import logging
import os
import tensorflow as tf


class Tokenizer(abc.ABC):
    """Abstract tokenizer."""

    def __init__(self, config):
        self.default_config = self._get_default_config()
        if config:
            self.default_config.update(config)
        self.config = self.default_config

        self.vocab_size_exclude_special_tokens = None
        self._tokens2ids_dict = {}
        self._ids2tokens_dict = {}
        self.tokens2ids_table = None
        self.ids2tokens_table = None

    def tokenize(self, corpus_files):
        for f in corpus_files:
            if not os.path.exists(f):
                logging.warning('File %s does not exist.' % f)
                continue
            with open(f, mode='rt', encoding='utf8') as fin:
                for line in fin:
                    line = line.strip('\n').strip()
                    if not line:
                        continue
                    self._process_line(line)

        assert len(self._tokens2ids_dict.keys()) == len(self._ids2tokens_dict.keys())
        self.vocab_size_exclude_special_tokens = len(self._tokens2ids_dict.keys())
        self._tokens2ids_dict[self.unk_token] = self.unk_id
        self._tokens2ids_dict[self.sos_token] = self.sos_id
        self._tokens2ids_dict[self.eos_token] = self.eos_id
        self._tokens2ids_dict[self.cls_token] = self.cls_id
        self._tokens2ids_dict[self.sep_token] = self.sep_id
        self._tokens2ids_dict[self.mask_token] = self.mask_id
        self._ids2tokens_dict[self.unk_id] = self.unk_token
        self._ids2tokens_dict[self.sos_id] = self.sos_token
        self._ids2tokens_dict[self.eos_id] = self.eos_token
        self._ids2tokens_dict[self.cls_id] = self.cls_token
        self._ids2tokens_dict[self.sep_id] = self.sep_token
        self._ids2tokens_dict[self.mask_id] = self.mask_token

    def _process_line(self, line):
        raise NotImplementedError()

    def initialize_lookup_tables(self):
        if self.tokens2ids_table is None:
            token2id_initializer = tf.lookup.KeyValueTensorInitializer(
                keys=list(self._tokens2ids_dict.keys()),
                values=list(self._tokens2ids_dict.values()),
                key_dtype=tf.dtypes.string,
                value_dtype=tf.dtypes.int64)
            self.tokens2ids_table = tf.lookup.StaticHashTable(
                initializer=token2id_initializer,
                default_value=self.unk_id,
                name='token2id_table')
        if self.ids2tokens_table is None:
            id2token_initializer = tf.lookup.KeyValueTensorInitializer(
                keys=list(self._ids2tokens_dict.keys()),
                values=list(self._ids2tokens_dict.values()),
                key_dtype=tf.dtypes.int64,
                value_dtype=tf.dtypes.string)
            self.ids2tokens_table = tf.lookup.StaticHashTable(
                initializer=id2token_initializer,
                default_value=self.unk_token,
                name='id2token_table')

    def tokens2ids(self, tokens):
        """Convert tokens to ids.

        Args:
            tokens: A string tensor

        Returns:
            A int tensor
        """
        return self.tokens2ids_table.lookup(tokens)

    def ids2tokens(self, ids):
        """Convert ids to tokens.

        Args:
            ids: A int tensor

        Returns:
            A string tensor
        """
        return self.ids2tokens_table.lookup(ids)

    def save_vocab(self, output_file):
        """Save vocab words(include special tokens in the end) to a file."""
        with open(output_file, mode='wt', encoding='utf8') as fout:
            for k in self._tokens2ids_dict.keys():
                fout.write(k + '\n')
        logging.info("Saved vocab to %s." % fout)

    @property
    def tokens2ids_dict(self):
        return self._tokens2ids_dict

    @property
    def ids2tokens_dict(self):
        return self._ids2tokens_dict

    @property
    def vocab_size(self):
        if not self.vocab_size_exclude_special_tokens:
            raise ValueError('vocab_size_exclude_special_tokens not intialized!')
        return self.vocab_size_exclude_special_tokens + len(self.default_config.keys())

    @property
    def unk_token(self):
        return self.config.get('unk_token', '<UNK>')

    @property
    def unk_id(self):
        return self.vocab_size_exclude_special_tokens

    @property
    def sos_token(self):
        return self.config.get('sos_token', '<SOS>')

    @property
    def sos_id(self):
        return self.vocab_size_exclude_special_tokens + 1

    @property
    def eos_token(self):
        return self.config.get('eos_token', '<EOS>')

    @property
    def eos_id(self):
        return self.vocab_size_exclude_special_tokens + 2

    @property
    def cls_token(self):
        return self.config.get('cls_token', '[CLS]')

    @property
    def cls_id(self):
        return self.vocab_size_exclude_special_tokens + 3

    @property
    def sep_token(self):
        return self.config.get('sep_token', '[SEP]')

    @property
    def sep_id(self):
        return self.vocab_size_exclude_special_tokens + 4

    @property
    def mask_token(self):
        return self.config.get('mask_token', '[MASK]')

    @property
    def mask_id(self):
        return self.vocab_size_exclude_special_tokens + 5

    @staticmethod
    def _get_default_config():
        c = {
            'unk_token': '<UNK>',
            'sos_token': '<SOS>',
            'eos_token': '<EOS>',
            'cls_token': '[CLS]',
            'sep_token': '[SEP]',
            'mask_token': '[MASK]'
        }
        return c
