import tensorflow as tf

from .abstract_dataset_builder import AbstractDatasetBuilder


class BertTFRecordDatasetBuilder(AbstractDatasetBuilder):

    def __init__(self, max_sequence_length=512, max_predictions_per_seq=20, record_option=None, **kwargs):
        super().__init__(**kwargs)
        self.record_option = record_option
        self.max_sequence_length = max_sequence_length
        self.max_predictions_per_seq = max_predictions_per_seq

    def build_train_dataset(
            self,
            record_files,
            batch_size=32,
            repeat_count=1,
            buffer_size=1000000,
            seed=None,
            reshuffle=True,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            dropout_remainder=False,
            **kwargs):
        dataset = tf.data.TFRecordDataset(record_files, compression_type=self.record_option)
        dataset = dataset.repeat(repeat_count)
        dataset = dataset.shuffle(
            buffer_size=buffer_size,
            seed=seed,
            reshuffle_each_iteration=reshuffle)
        dataset = dataset.map(self._parse_example_fn, num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(
            batch_size, drop_remainder=dropout_remainder
        ).prefetch(batch_size)
        dataset = dataset.map(lambda x, y, z, p, l: ((x, y, z), (p, l)))
        return dataset

    def build_valid_dataset(
            self,
            record_files,
            batch_size=32,
            repeat_count=1,
            buffer_size=1000000,
            seed=None,
            reshuffle=True,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            dropout_remainder=False,
            **kwargs):
        return self.build_train_dataset(
            record_files=record_files,
            batch_size=batch_size,
            repeat_count=repeat_count,
            buffer_size=buffer_size,
            seed=seed,
            reshuffle=reshuffle,
            num_parallel_calls=num_parallel_calls,
            dropout_remainder=dropout_remainder,
            **kwargs)

    def build_predict_dataset(self, record_files, batch_size=1, repeat_count=1, dropout_remainder=False, **kwargs):
        dataset = tf.data.TFRecordDataset(record_files, compression_type=self.record_option)
        dataset = dataset.repeat(repeat_count)
        dataset = dataset.map(self._parse_example_fn)
        dataset = dataset.batch(
            batch_size, drop_remainder=dropout_remainder
        ).prefetch(batch_size)
        dataset = dataset.map(lambda x, y, z, p, l: ((x, y, z), None))
        return dataset

    def _parse_example_fn(self, record):
        """Parse tfrecord to example. Change this parse process according to your record files format."""
        MAX_SEQ_LEN = self.max_sequence_length
        MAX_PREDICTIONS_PER_SEQ = self.max_predictions_per_seq
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
        # features = {
        #     'input_ids': example.get('input_ids', None),
        #     'input_mask': example.get('input_mask', None),
        #     'segment_ids': example.get('segment_ids', None),
        # }
        # labels = {
        #     'predictions': example['original_ids'],
        #     'relations': tf.one_hot(example['next_sentence_labels'], 2)
        # }
        input_ids = example['input_ids']
        input_mask = example['input_mask']
        segment_ids = example['segment_ids']
        predictions = example['original_ids']
        relations = example['next_sentence_labels']
        return (input_ids, input_mask, segment_ids, predictions, relations)
