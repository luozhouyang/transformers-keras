import logging
import time

import tensorflow as tf

from transformers_keras.datasets.transformer_dataset import TransformerDataset
from transformers_keras.runners import AbstractRunner
from transformers_keras.tokenizers.space_tokenizer import SpaceTokenizer
from transformers_keras.transformer import funcs
from transformers_keras.transformer.custom_lr_schedule import CustomLearningRateSchedule
from transformers_keras.transformer.transformer_model import Transformer


class TransformerRunner(AbstractRunner):

    def __init__(self, config=None):
        super(TransformerRunner, self).__init__(config)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.eval_loss = tf.keras.metrics.Mean(name='train_loss')
        self.eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @staticmethod
    def loss_fun(real, pred):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]  # sos + ids
        tar_real = tar[:, 1:]  # ids + eos
        enc_padding_mask, combined_mask, dec_padding_mask = funcs.create_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            predictions, _, _ = self.model(
                inputs=(inp, tar_inp),
                training=True,
                mask=(enc_padding_mask, combined_mask, dec_padding_mask))
            loss = self.loss_fun(tar_real, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)

    @tf.function
    def eval_step(self, inp, tar):
        tar_inp = tar[:, :-1]  # sos + ids
        tar_real = tar[:, 1:]  # ids + eos
        enc_padding_mask, combined_mask, dec_padding_mask = funcs.create_masks(inp, tar_inp)
        predictions, _, _ = self.model(
            inputs=(inp, tar_inp),
            training=False,
            mask=(enc_padding_mask, combined_mask, dec_padding_mask))
        loss = self.loss_fun(tar_real, predictions)
        self.eval_loss(loss)
        self.eval_accuracy(tar_real, predictions)

    def _create_src_tokenizer(self):
        return SpaceTokenizer(self.config)

    def _create_tgt_tokenizer(self):
        return SpaceTokenizer(self.config)

    def train(self):
        self._init_src_tokenizer()
        self._init_tgt_tokenizer()
        self.model = self._build_model()
        self.ckpt_manager = self._build_ckpt_manager()
        train_files = (self.config['train_src_files'], self.config['train_tgt_files'])
        dataset = TransformerDataset(
            src_tokenizer=self.src_tokenizer, tgt_tokenizer=self.tgt_tokenizer, config=self.config)
        train_dataset = dataset.build_train_dataset(train_files)
        for epoch in range(self.config.get('epochs', 10)):
            start = time.time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for (batch, (inp, tar)) in enumerate(train_dataset):
                self.train_step(inp, tar)

                if batch % 100 == 0:
                    logging.info('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, self.train_loss.result(), self.train_accuracy.result()))

            if (epoch + 1) % 2 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                logging.info('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

            logging.info('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, self.train_loss.result(), self.train_accuracy.result()))

            logging.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    def train_and_evaluate(self):
        self._init_src_tokenizer()
        self._init_tgt_tokenizer()
        self.model = self._build_model()
        self.ckpt_manager = self._build_ckpt_manager()
        train_files = (self.config['train_src_files'], self.config['train_tgt_files'])
        dataset = TransformerDataset(
            src_tokenizer=self.src_tokenizer, tgt_tokenizer=self.tgt_tokenizer, config=self.config)
        train_dataset = dataset.build_train_dataset(train_files)

        for epoch in range(self.config.get('epochs', 10)):
            start = time.time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for (batch, (inp, tar)) in enumerate(train_dataset):
                self.train_step(inp, tar)

                if batch % 1 == 0:
                    logging.info('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, self.train_loss.result(), self.train_accuracy.result()))

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                logging.info('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

                logging.info('Starting to evaluating model...')
                self.evaluate()
                logging.info('Finished to evaluate model. Continue training...\n')

            logging.info('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, self.train_loss.result(), self.train_accuracy.result()))

            logging.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    def evaluate(self):
        self._init_src_tokenizer()
        self._init_tgt_tokenizer()
        self.model = self._build_model()
        self.ckpt_manager = self._build_ckpt_manager()
        train_files = (self.config['train_src_files'], self.config['train_tgt_files'])
        dataset = TransformerDataset(
            src_tokenizer=self.src_tokenizer, tgt_tokenizer=self.tgt_tokenizer, config=self.config)
        eval_dataset = dataset.build_eval_dataset(train_files)

        self.eval_loss.reset_states()
        self.eval_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(eval_dataset):
            self.eval_step(inp, tar)
        logging.info('Evaluation finished. Loss {:.4f} Accuracy {:.4f}'.format(
            self.eval_loss.result(), self.eval_accuracy.result()))

    def predict(self):
        self._init_src_tokenizer()
        self._init_tgt_tokenizer()
        self.model = self._build_model()
        self.ckpt_manager = self._build_ckpt_manager()
        predict_files = (self.config['predict_src_files'])
        dataset = TransformerDataset(
            src_tokenizer=self.src_tokenizer, tgt_tokenizer=self.tgt_tokenizer, config=self.config)
        predict_dataset = dataset.build_predict_dataset(predict_files)
        tgt_sos = tf.constant([self.tgt_tokenizer.sos_id])
        for (batch, (inp)) in enumerate(predict_dataset):
            b = tf.shape(inp)[0]
            output = tf.tile(tgt_sos, [tf.shape(inp)[0], ])
            output = tf.expand_dims(output, -1)
            output = tf.cast(output, tf.dtypes.int64)
            enc_attn = None
            dec_attn = None
            early_stop = False
            for i in range(self.config.get('predict_max_steps', 10)):
                enc_padding_mask, combined_mask, dec_padding_mask = funcs.create_masks(inp, output)
                predictions, enc_attn, dec_attn = self.model(
                    inputs=(inp, output),
                    training=False,
                    mask=(enc_padding_mask, combined_mask, dec_padding_mask))
                predictions = predictions[:, -1:, :]
                predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.dtypes.int64)
                if predicted_id == self.tgt_tokenizer.eos_id:
                    # result = tf.cast(tf.squeeze(output, axis=0), tf.dtypes.int64)
                    result = tf.squeeze(output, axis=0)
                    logging.info('batch {} results: {}'.format(batch, self.tgt_tokenizer.decode(result)))
                    early_stop = True
                    break
                output = tf.concat([output, predicted_id], axis=-1)
            if not early_stop:
                result = tf.squeeze(output, axis=0)
                logging.info('batch {} results: {}'.format(batch, self.tgt_tokenizer.decode(result)))

    def export(self):
        pass

    def _build_model(self):
        if self.model:
            return self.model
        model = Transformer(
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            d_model=self.config['d_model'],
            dff=self.config['dff'],
            input_vocab_size=self.src_tokenizer.vocab_size,
            target_vocab_size=self.tgt_tokenizer.vocab_size,
            rate=self.config['rate'])
        logging.info('Created new transformer model.')
        return model

    def _build_optimizer(self):
        lr = CustomLearningRateSchedule(self.config['d_model'])
        optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        return optimizer

    def _build_ckpt(self):
        ckpt = tf.train.Checkpoint(transformer=self.model, optimizer=self.optimizer)
        return ckpt

    def _get_default_config(self):
        parent = super(TransformerRunner, self)._get_default_config()
        parent.update({
            'd_model': 512,
            'num_heads': 8,
            'num_layers': 6,
            'dff': 2048,
            'rate': 0.1
        })
        return parent


if __name__ == '__main__':
    config = {
        'train_src_files': ['testdata/train.src.txt'],
        'train_tgt_files': ['testdata/train.tgt.txt'],
        'eval_src_files': ['testdata/train.src.txt'],
        'eval_tgt_files': ['testdata/train.tgt.txt'],
        'predict_src_files': ['testdata/train.src.txt'],
    }
    r = TransformerRunner(config)
    r.train()
