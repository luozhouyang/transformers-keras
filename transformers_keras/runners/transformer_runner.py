import logging
import os
import time

import tensorflow as tf

from transformers_keras.datasets.transformer_dataset import TransformerDataset
from transformers_keras.tokenizers.space_tokenizer import SpaceTokenizer
from transformers_keras.transformer import funcs
from transformers_keras.transformer.custom_lr_schedule import CustomLearningRateSchedule
from transformers_keras.transformer.transformer_model import Transformer


class TransformerRunner:

    def __init__(self, config=None):
        default_config = self.get_default_config()
        if config:
            default_config.update(config)
        self.config = default_config
        lr = CustomLearningRateSchedule(self.config['d_model'])
        self.optimizer = tf.keras.optimizers.Adam(
            lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        self.ckpt_path = self.config.get('ckpt_path', '/tmp/models/transformer')
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        # persist token <-> id map to a file, so we can ensure the map relations never change
        self.src_vocab_path = os.path.join(self.ckpt_path, 'vocab.src.txt')
        self.tgt_vocab_path = os.path.join(self.ckpt_path, 'vocab.tgt.txt')

        self.src_tokenizer = None
        self.tgt_tokenizer = None

        self.model = None
        self.ckpt_manager = None

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    def _init_tokenizer(self):
        def concat_files(names):
            files = []
            for n in names:
                l = self.config.get(n, [])
                if not l:
                    continue
                for f in l:
                    files.append(f)
            return files

        if not self.src_tokenizer:
            self.src_tokenizer = SpaceTokenizer(self.config)

            if os.path.exists(self.src_vocab_path):
                logging.info('Starting to build src tokenizer from vocab...')
                self.src_tokenizer.build_from_vocab(self.src_vocab_path)
                logging.info('Finished to build src tokenizer from vocab.')
                logging.info('src language vocab size: %d\n' % self.src_tokenizer.vocab_size)
            else:
                logging.info('Starting to build src tokenizer from corpus...')
                train_files = concat_files(['train_src_files', 'eval_src_files'])
                self.src_tokenizer.build_from_corpus(train_files)
                self.src_tokenizer.save_to_vocab(self.src_vocab_path)
                logging.info('Finished to build src tokenizer from corpus.')
                logging.info('Saved src vocab to: %s' % self.src_vocab_path)
                logging.info('src language vocab size: %d\n' % self.src_tokenizer.vocab_size)

        if not self.tgt_tokenizer:
            self.tgt_tokenizer = SpaceTokenizer(self.config)

            if os.path.exists(self.tgt_vocab_path):
                logging.info('Starting to build tgt tokenizer from vocab...')
                self.tgt_tokenizer.build_from_vocab(self.tgt_vocab_path)
                logging.info('Finished to build tgt tokenizer from vocab.')
                logging.info('tgt language vocab size: %d\n' % self.tgt_tokenizer.vocab_size)
            else:
                logging.info('Starting to build tgt tokenizer from corpus...')
                train_files = concat_files(['train_tgt_files', 'eval_tgt_files'])
                self.tgt_tokenizer.build_from_corpus(train_files)
                self.tgt_tokenizer.save_to_vocab(self.tgt_vocab_path)
                logging.info('Finished to build tgt tokenizer from corpus.')
                logging.info('Saved tgt vocab to: %s' % self.tgt_vocab_path)
                logging.info('tgt language vocab size: %d\n' % self.tgt_tokenizer.vocab_size)

    def _create_model(self):
        self.model = Transformer(
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            d_model=self.config['d_model'],
            dff=self.config['dff'],
            input_vocab_size=self.src_tokenizer.vocab_size,
            target_vocab_size=self.tgt_tokenizer.vocab_size,
            rate=self.config['rate'])

        # init checkpoints
        ckpt = tf.train.Checkpoint(
            transformer=self.model, optimizer=self.optimizer)

        logging.info("Checkpoints will be saved in %s" % self.ckpt_path)
        self.ckpt_manager = tf.train.CheckpointManager(
            ckpt, self.ckpt_path, max_to_keep=config.get('max_keep_ckpt', 10))
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            logging.info("Restore ckpt from: %s", self.ckpt_manager.latest_checkpoint)

    def train(self):

        def loss_fun(real, pred):
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = loss_object(real, pred)
            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask
            return tf.reduce_mean(loss_)

        @tf.function
        def train_step(inp, tar):
            tar_inp = tar[:, :-1]  # sos + ids
            tar_real = tar[:, 1:]  # ids + eos
            enc_padding_mask, combined_mask, dec_padding_mask = funcs.create_masks(inp, tar_inp)
            with tf.GradientTape() as tape:
                predictions, _, _ = self.model(
                    inputs=(inp, tar_inp),
                    training=True,
                    mask=(enc_padding_mask, combined_mask, dec_padding_mask))
                loss = loss_fun(tar_real, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            self.train_loss(loss)
            self.train_accuracy(tar_real, predictions)

        self._init_tokenizer()
        self._create_model()

        dataset = TransformerDataset(self.src_tokenizer, self.tgt_tokenizer, self.config)
        train_files = (self.config['train_src_files'], self.config['train_tgt_files'])
        train_dataset = dataset.build_train_dataset(train_files)

        for epoch in range(self.config.get('epochs', 10)):
            start = time.time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for (batch, (inp, tar)) in enumerate(train_dataset):
                train_step(inp, tar)

                if batch % 1 == 0:
                    logging.info('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, self.train_loss.result(), self.train_accuracy.result()))

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                logging.info('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

            logging.info('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, self.train_loss.result(), self.train_accuracy.result()))

            logging.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    def eval(self, input_sentence):
        pass

    def predict(self):
        pass

    def export(self):
        pass

    @staticmethod
    def get_default_config():
        c = {
            'd_model': 512,
            'num_heads': 8,
            'num_layers': 6,
            'dff': 2048,
            'rate': 0.1
        }
        return c


if __name__ == '__main__':
    config = {
        'train_src_files': ['testdata/train.src.txt'],
        'train_tgt_files': ['testdata/train.tgt.txt'],
    }
    r = TransformerRunner(config)
    r.train()
