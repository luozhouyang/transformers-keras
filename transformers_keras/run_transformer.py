import argparse
import json
import logging
import os

import tensorflow as tf

from transformers_keras.callbacks import (ModelStepCheckpoint,
                                          SavedModelExporter)

from .losses import MaskedSparseCategoricalCrossentropy
from .metrics import MaskedSparseCategoricalAccuracy
from .modeling_transformer import *


def build_model(config):
    x = tf.keras.layers.Input(shape=(config.max_positions,), dtype=tf.int32, name='x')
    y = tf.keras.layers.Input(shape=(config.max_positions,), dtype=tf.int32, name='y')
    transformer = Transformer(config)
    logits, enc_attns, dec_attns_0, dec_attens_1 = transformer(inputs=(x, y))
    probs = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x), name='probs')(logits)

    model = tf.keras.Model(inputs=[x, y], outputs=[probs])

    model.compile(
        optimizer='adam',  # TODO(zhouyang.luo) schedule learning rate
        loss={
            'probs': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        },
        metrics={
            'probs': [
                tf.keras.metrics.SparseCategoricalAccuracy(),
            ]
        }
    )

    return model


class TransformerDataConfig(object):

    def __init__(self, **kwargs):
        super().__init__()
        self.record_option = kwargs.get('record_option', 'GZIP')
        self.skip_count = kwargs.get('skip_count', 0)
        self.repeat = kwargs.get('repeat', 1)
        self.shuffle_buffer_size = kwargs.get('shuffle_buffer_size', 1000000)
        self.shuffle_seed = kwargs.get('shuffle_seed', None)
        self.reshuffle_each_iteration = kwargs.get('reshuffle_each_iteration', True)
        self.train_batch_size = kwargs.get('train_batch_size', 32)
        self.valid_batch_size = kwargs.get('valid_batch_size', 32)
        self.predict_batch_size = kwargs.get('predict_batch_size', 1)
        self.drop_remainder = kwargs.get('drop_remainder', True)

        self.ckpt_steps = kwargs.get('ckpt_steps', 10000)
        self.ckpt_max_nums = kwargs.get('ckpt_max_nums', 10)
        self.export_steps = kwargs.get('export_steps', 10000)
        self.epochs = kwargs.get('epochs', 10)

        self.model_dir = kwargs.get('model_dir', '/tmp/transformer')
        self.train_record_files = kwargs.get('train_record_files', None)
        self.valid_record_files = kwargs.get('valid_record_files', None)
        self.predict_record_files = kwargs.get('predict_record_files', None)


class TransformerDataset(object):

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

    def _parse_example(self, record):
        name_to_features = {
            'src_ids': tf.io.FixedLenFeature([16], tf.int64),
            'tgt_ids': tf.io.FixedLenFeature([16], tf.int64)
        }

        example = tf.io.parse_single_example(record, name_to_features)
        features = example['src_ids']
        labels = example['tgt_ids']
        inputs = {
            'x': example['src_ids'],
            'y': example['tgt_ids']
        }
        return inputs

    def build_train_dataset(self, train_record_files):
        dataset = tf.data.Dataset.from_tensor_slices(train_record_files)
        if self.config.skip_count > 0:
            dataset = dataset.interleave(
                lambda x: tf.data.TFRecordDataset(
                    x, compression_type=self.config.record_option).skip(self.config.skip_count),
                cycle_length=len(train_record_files))
        else:
            dataset = dataset.interleave(
                lambda x: tf.data.TFRecordDataset(x, compression_type=self.config.record_option),
                cycle_length=len(train_record_files))
        dataset = dataset.repeat(self.config.repeat)
        dataset = dataset.shuffle(
            buffer_size=self.config.shuffle_buffer_size,
            seed=self.config.shuffle_seed,
            reshuffle_each_iteration=self.config.reshuffle_each_iteration
        )
        dataset = dataset.map(lambda x: self._parse_example(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(
            self.config.train_batch_size,
            drop_remainder=self.config.drop_remainder
        ).prefetch(self.config.train_batch_size)
        return dataset

    def build_valid_dataset(self, valid_record_files):
        if not valid_record_files:
            logging.warning('valid_record_files in None or empty.')
            return None
        dataset = tf.data.Dataset.from_tensor_slices(valid_record_files)
        if self.config.skip_count > 0:
            dataset = dataset.interleave(
                lambda x: tf.data.TFRecordDataset(
                    x, compression_type=self.config.record_option).skip(self.config.skip_count),
                cycle_length=len(valid_record_files))
        else:
            dataset = dataset.interleave(
                lambda x: tf.data.TFRecordDataset(x, compression_type=self.config.record_option),
                cycle_length=len(valid_record_files))
        dataset = dataset.map(lambda x: self._parse_example(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(
            self.config.valid_batch_size,
            drop_remainder=False
        ).prefetch(self.config.valid_batch_size)
        return dataset

    def build_predict_dataset(self, predict_record_files):
        dataset = tf.data.Dataset.from_tensor_slices(predict_record_files)
        if self.config.skip_count > 0:
            dataset = dataset.interleave(
                lambda x: tf.data.TFRecordDataset(
                    x, compression_type=self.config.record_option).skip(self.config.skip_count),
                cycle_length=len(predict_record_files))
        else:
            dataset = dataset.interleave(
                lambda x: tf.data.TFRecordDataset(x, compression_type=self.config.record_option),
                cycle_length=len(predict_record_files))
        dataset = dataset.map(lambda x: self._parse_example(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(
            self.config.predict_batch_size,
            drop_remainder=False
        ).prefetch(self.config.predict_batch_size)
        return dataset


def train(model, dataconfig):
    dataset = TransformerDataset(dataconfig)
    train_dataset = dataset.build_train_dataset(dataconfig.train_record_files)
    valid_dataset = dataset.build_valid_dataset(dataconfig.valid_record_files)
    model_dir = dataconfig.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    ckpt_path = model_dir + '/' + 'transformer-{epoch:04d}.ckpt'
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        validation_steps=None,
        epochs=dataconfig.epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
            tf.keras.callbacks.TensorBoard(os.path.join(model_dir, 'tensorboard'), update_freq='batch'),
            ModelStepCheckpoint(
                model,
                model_dir,
                every_steps=dataconfig.ckpt_steps,
                max_keep_ckpt=dataconfig.ckpt_max_nums),
            SavedModelExporter(
                model,
                os.path.join(model_dir, 'export'),
                every_steps=dataconfig.export_steps),
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'predict'])
    parser.add_argument('--model_config', type=str, help='Model config file in JSON format.')
    parser.add_argument('--data_config', type=str, help='Data config file in JSON format.')

    args, _ = parser.parse_known_args()

    model_config, data_config = {}, {}
    if os.path.exists(args.model_config):
        logging.info('Load model config from: %s.' % args.model_config)
        with open(args.model_config, mode='rt', encoding='utf8') as f:
            model_config = json.load(f)
    if os.path.exists(args.data_config):
        logging.info('Load data config from: %s.' % args.data_config)
        with open(args.data_config, mode='rt', encoding='utf8') as f:
            data_config = json.load(f)

    model_config = TransformerConfig(**model_config)
    model = build_model(model_config)

    data_config = TransformerDataConfig(**data_config)

    model.summary()

    if 'train' == args.mode:
        train(model, data_config)

    else:
        raise ValueError('Invalid `mode`: %s' % args.mode)
