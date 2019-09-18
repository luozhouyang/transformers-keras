import logging

import time

import tensorflow as tf

from transformer import funcs
from transformer.custom_lr_schedule import CustomLearningRateSchedule
from transformer.transformer_model import Transformer
from datasets.nlp.xy_dataset import XYSameFileDataset

config = {
    "num_layers": 2,
    "d_model": 128,
    "dff": 512,
    "num_heads": 8,
    "input_vocab_size": 1000,
    "target_vocab_size": 1000,
    "dropout": 0.1,
}

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')


class Runner:

    def __init__(self, config):
        self.config = config

        lr = CustomLearningRateSchedule(config['d_model'])
        self.optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.model = Transformer(
            num_layers=config['num_layers'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            input_vocab_size=config['input_vocab_size'],
            target_vocab_size=config['target_vocab_size'],
            dff=config['dff'],
            rate=config['dropout'])

        ckpt = tf.train.Checkpoint(transformer=self.model, optimizer=self.optimizer)
        ckpt_path = config.get('ckpt_path', '/tmp/transformer/models')
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=config.get('max_keep_ckpt', 10))
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            logging.info("Restore ckpt from: %s", self.ckpt_manager.latest_checkpoint)

    def train(self, train_dataset):
        model = self.model
        optimizer = self.optimizer

        train_step_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        ]

        @tf.function(input_signature=train_step_signature)
        def train_step(inp, tar):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            enc_padding_mask, combined_mask, dec_padding_mask = funcs.create_masks(inp, tar_inp)

            with tf.GradientTape() as tape:
                predictions, _ = model(
                    inputs=(inp, tar_inp),
                    training=True,
                    mask=(enc_padding_mask, combined_mask, dec_padding_mask))
                loss = loss_function(tar_real, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            train_accuracy(tar_real, predictions)

        for epoch in range(self.config.get('epochs', 1)):
            start = time.time()

            train_loss.reset_states()
            train_accuracy.reset_states()

            # inp -> portuguese, tar -> english
            for (batch, (inp, tar)) in enumerate(train_dataset):
                logging.info("input shape: %s" % tf.shape(inp))
                train_step(inp, tar)

                if batch % 500 == 0:
                    logging.info('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, train_loss.result(), train_accuracy.result()))

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                logging.info('Saving checkpoint for epoch {} at {}'.format(
                    epoch + 1, ckpt_save_path))

            logging.info('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, train_loss.result(), train_accuracy.result()))

            logging.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    def eval(self):
        pass

    def predict(self):
        pass

    def export(self):
        pass


if __name__ == '__main__':
    runner = Runner(config)
    dataset_config = {
        'x_vocab_file': 'testdata/vocab_src.txt',
        'y_vocab_file': 'testdata/vocab_tgt.txt'
    }
    dataset = XYSameFileDataset(config=dataset_config, logger_name='root')
    train_dataset = dataset.build_train_dataset(train_files=['testdata/train.txt'])
    logging.info("Build train dataset successfully.")
    runner.train(train_dataset)
