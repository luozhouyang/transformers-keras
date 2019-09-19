import logging
import time

import tensorflow as tf
from nlp_datasets.nlp import XYZSameFileDataset

from transformers_keras.runners.runner import Runner
from transformers_keras.transformer import funcs
from transformers_keras.transformer.custom_lr_schedule import CustomLearningRateSchedule
from transformers_keras.transformer.transformer_model import Transformer


class TransformerRunner(Runner):

    def __init__(self, config=None):
        default_config = self.get_default_config()
        if config:
            default_config.update(config)
        self.config = default_config
        lr = CustomLearningRateSchedule(self.config['d_model'])
        self.optimizer = tf.keras.optimizers.Adam(
            lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.model = Transformer(
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            d_model=self.config['d_model'],
            dff=self.config['dff'],
            input_vocab_size=self.config['input_vocab_size'],
            target_vocab_size=self.config['target_vocab_size'],
            rate=self.config['rate'])

        # init checkpoints
        ckpt = tf.train.Checkpoint(
            transformer=self.model, optimizer=self.optimizer)
        ckpt_path = config.get('ckpt_path', '/tmp/models/transformer')
        self.ckpt_manager = tf.train.CheckpointManager(
            ckpt, ckpt_path, max_to_keep=config.get('max_keep_ckpt', 10))
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            logging.info("Restore ckpt from: %s", self.ckpt_manager.latest_checkpoint)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')

        self.dataset = XYZSameFileDataset(self.config)

    def train(self):

        def loss_fun(real, pred):
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = loss_object(real, pred)
            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask
            return tf.reduce_mean(loss_)

        def train_step(inp, tar):
            tar_inp = tar[:, :-1]  # sos + ids
            tar_real = tar[:, 1:]  # ids + eos
            enc_padding_mask, combined_mask, dec_padding_mask = funcs.create_masks(inp, tar_inp)
            with tf.GradientTape() as tape:
                predictions, _ = self.model(
                    inputs=(inp, tar_inp),
                    training=True,
                    mask=(enc_padding_mask, combined_mask, dec_padding_mask))
                loss = loss_fun(tar_real, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            self.train_loss(loss)
            self.train_accuracy(tar_real, predictions)

        train_dataset = self.dataset.build_train_dataset(self.config['train_files'])

        for epoch in range(self.config.get('epochs', 10)):
            start = time.time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for (batch, (inp, tar)) in enumerate(train_dataset):
                logging.info("input shape: %s" % tf.shape(inp))
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

    def eval(self):
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
            'input_vocab_size': 100,
            'target_vocab_size': 100,
            'rate': 0.1
        }
        return c


if __name__ == '__main__':
    config = {
        'train_files': ['testdata/train.txt'],
        'x_vocab_file': 'testdata/vocab_src.txt',
        'y_vocab_file': 'testdata/vocab_tgt.txt'
    }
    r = TransformerRunner(config)
    r.train()
