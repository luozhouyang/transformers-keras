import argparse
import json
import logging
import os

import tensorflow as tf

from transformers_keras.callbacks.model_step_checkpoint import \
    ModelStepCheckpoint
from transformers_keras.callbacks.saved_model_exporter import \
    SavedModelExporter

from .losses import MaskedSparseCategoricalCrossentropy
from .metrics import MaskedSparseCategoricalAccuracy
from .modeling_bert import Bert4PreTraining, BertConfig


def build_model(config):
    input_ids = tf.keras.layers.Input(shape=(config.max_sequence_length,), dtype=tf.int32, name='input_ids')
    input_mask = tf.keras.layers.Input(shape=(config.max_sequence_length,), dtype=tf.int32, name='input_mask')
    segment_ids = tf.keras.layers.Input(shape=(config.max_sequence_length,), dtype=tf.int32, name='segment_ids')

    outputs = Bert4PreTraining(config, name='bert')(inputs=(input_ids, segment_ids, input_mask))

    predictions = tf.keras.layers.Lambda(lambda x: x, name='predictions')(outputs[0])
    relations = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x), name='relations')(outputs[1])
    attentions = tf.keras.layers.Lambda(lambda x: x, name='attentions')(outputs[2])

    model = tf.keras.Model(inputs=[input_ids, segment_ids, input_mask], outputs=[predictions, relations, attentions])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-8, clipnorm=1.0),
        loss={
            'predictions': MaskedSparseCategoricalCrossentropy(mask_id=0, from_logits=True),
            'relations': tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        },
        metrics={
            'predictions': [
                MaskedSparseCategoricalAccuracy(mask_id=0, from_logits=False),
            ],
            'relations': [
                tf.keras.metrics.CategoricalAccuracy(),
            ]
        })
    return model


class BertDataConfig(object):

    def __init__(self, **kwargs):
        super().__init__()
        self.train_batch_size = kwargs.pop('train_batch_size', 32)
        self.train_record_files = kwargs.pop('train_record_files', None)
        self.valid_batch_size = kwargs.pop('valid_batch_size', 32)
        self.valid_record_files = kwargs.pop('valid_record_files', None)
        self.predict_batch_size = kwargs.pop('predict_batch_size', 1)
        self.shuffle_buffer_size = kwargs.pop('shuffle_buffer_size', 1000000)
        self.shuffle_seed = kwargs.pop('shuffle_seed', None)
        self.reshuffle_each_iteration = kwargs.pop('reshuffle_each_iteration', True)
        self.drop_remainder = kwargs.pop('drop_remainder', True)
        self.epochs = kwargs.pop('epochs', 10)
        self.model_dir = kwargs.pop('model_dir', '/tmp/bert')
        self.record_option = kwargs.pop('record_option', 'GZIP')
        self.save_ckpt_steps = kwargs.pop('save_ckpt_steps', 10000)
        self.max_ckpt_nums = kwargs.pop('max_ckpt_nums', 10)
        self.export_steps = kwargs.pop('export_steps', 100000)

    @classmethod
    def from_json_file(cls, filename):
        d = {}
        if not os.path.exists(filename):
            logging.warning('Config file %s does not exists.' % filename)
            return cls(**d)
        logging.info('Load data config from: %s.' % filename)
        with open(filename, mode='rt', encoding='utf8') as fin:
            d = json.load(fin)
        return cls(**d)


class BertDataset(object):

    def __init__(self, config, **kwargs):
        self.config = config

    def _parse_example_fn(self, record):
        raise NotImplementedError()

    def build_train_dataset(self, train_record_files):
        dataset = tf.data.TFRecordDataset(train_record_files, compression_type=self.config.record_option)
        dataset = dataset.map(self._parse_example_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(
            buffer_size=self.config.shuffle_buffer_size,
            seed=self.config.shuffle_seed,
            reshuffle_each_iteration=self.config.reshuffle_each_iteration)
        dataset = dataset.batch(
            self.config.train_batch_size, drop_remainder=self.config.drop_remainder
        ).prefetch(self.config.train_batch_size)
        return dataset

    def build_valid_dataset(self, valid_record_files):
        if valid_record_files is None:
            return None
        dataset = tf.data.TFRecordDataset(valid_record_files)
        dataset = dataset.map(self._parse_example_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.config.valid_batch_size).prefetch(self.config.valid_batch_size)
        return dataset

    def build_predict_dataset(self, predict_record_files):
        dataset = tf.data.TFRecordDataset(predict_record_files)
        dataset = dataset.map(self._parse_example_fn)
        dataset = dataset.batch(self.config.predict_batch_size).prefetch(self.config.predict_batch_size)
        return dataset


class CustomBertDataset(BertDataset):

    def _parse_example_fn(self, record):
        """Parse tfrecord to example. Change this parse process according to your record files format."""
        MAX_SEQ_LEN = self.config.max_sequence_length
        MAX_PREDICTIONS_PER_SEQ = self.config.max_predictions_per_seq
        name_to_features = {
            'original_ids': tf.io.FixedLenFeature([MAX_SEQ_LEN], tf.int64),
            'input_ids': tf.io.FixedLenFeature([MAX_SEQ_LEN], tf.int64),
            'input_mask': tf.io.FixedLenFeature([MAX_SEQ_LEN], tf.int64),
            'segment_ids': tf.io.FixedLenFeature([MAX_SEQ_LEN], tf.int64),
            'masked_lm_positions': tf.io.FixedLenFeature([MAX_PREDICTIONS_PER_SEQ], tf.int64),
            'masked_lm_ids': tf.io.FixedLenFeature([MAX_PREDICTIONS_PER_SEQ], tf.int64),
            'masked_lm_weights': tf.io.FixedLenFeature([MAX_PREDICTIONS_PER_SEQ], tf.float32),
            'next_sentence_labels': tf.io.FixedLenFeature([1], tf.int64),
        }

        example = tf.io.parse_single_example(record, name_to_features)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t

        # extract needed values from example to matching network's inputs and outputs
        features = {
            'input_ids': example.get('input_ids', None),
            'input_mask': example.get('input_mask', None),
            'segment_ids': example.get('segment_ids', None),
        }
        labels = {
            'predictions': example['original_ids'],
            'relations': tf.one_hot(example['next_sentence_labels'], 2)
        }
        return (features, labels)


def train(model, dataconfig):
    dataset = CustomBertDataset(dataconfig)
    train_dataset = dataset.build_train_dataset(dataconfig.train_record_files)
    valid_dataset = dataset.build_valid_dataset(dataconfig.valid_record_files)

    model_dir = dataconfig.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    ckpt_path = model_dir + '/' + 'bert-{epoch:04d}.ckpt'
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        validation_steps=1000,
        epochs=dataconfig.epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='relations_loss', patience=5),
            tf.keras.callbacks.EarlyStopping(monitor='predictions_loss', patience=5),
            tf.keras.callbacks.TensorBoard(os.path.join(model_dir, 'tensorboard'), update_freq='batch'),
            ModelStepCheckpoint(
                model,
                model_dir,
                every_steps=dataconfig.save_ckpt_steps,
                max_keep_ckpt=dataconfig.max_ckpt_nums),
            SavedModelExporter(
                model,
                os.path.join(model_dir, 'export'),
                every_steps=dataconfig.export_steps)
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_config', type=str, help="Model config file in JSON format.")
    parser.add_argument('--data_config', type=str, help="Data config file in JSON format.")

    args, _ = parser.parse_known_args()

    bert_model_config = BertConfig.from_json_file(args.model_config)
    bert_data_config = BertDataConfig.from_json_file(args.data_config)
    model = build_model(bert_model_config)

    if 'train' == args.mode:
        train(model, bert_data_config)

    else:
        raise ValueError('Invalid `mode` argument.')
