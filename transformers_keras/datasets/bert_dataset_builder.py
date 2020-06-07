import tensorflow as tf

from .abstract_dataset_builder import AbstractDatasetBuilder


class BertTFRecordDatasetBuilder(AbstractDatasetBuilder):

    def __init__(self, max_sequence_length=512, max_predictions_per_seq=20, record_option=None, **kwargs):
        super().__init__(**kwargs)
        self.record_option = record_option
        self.max_sequence_length = max_sequence_length
        self.max_predictions_per_seq = max_predictions_per_seq

    def build_train_dataset(self, train_record_files, **kwargs):
        dataset = tf.data.TFRecordDataset(train_record_files, compression_type=self.record_option)
        dataset = dataset.repeat(self.train_repeat_count)
        dataset = dataset.shuffle(
            buffer_size=self.train_shuffle_buffer_size,
            seed=self.train_shuffle_seed,
            reshuffle_each_iteration=self.train_reshuffle_each_iteration)
        dataset = dataset.map(self._parse_example_fn, num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.batch(
            self.train_batch_size, drop_remainder=self.train_drop_remainder
        ).prefetch(self.prefetch_size)
        dataset = dataset.map(lambda x, y, z, p, l: ((x, y, z), (p, l)))
        return dataset

    def build_valid_dataset(self, valid_record_files, **kwargs):
        if valid_record_files is None:
            return None
        dataset = tf.data.TFRecordDataset(valid_record_files, compression_type=self.record_option)
        dataset = dataset.repeat(self.valid_repeat_count)
        dataset = dataset.shuffle(
            buffer_size=self.valid_shuffle_buffer_size,
            seed=self.valid_shuffle_seed,
            reshuffle_each_iteration=self.valid_reshuffle_each_iteration
        )
        dataset = dataset.map(self._parse_example_fn, num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.batch(
            self.valid_batch_size, drop_remainder=self.valid_drop_remainder
        ).prefetch(self.prefetch_size)
        dataset = dataset.map(lambda x, y, z, p, l: ((x, y, z), (p, l)))
        return dataset

    def build_predict_dataset(self, predict_record_files, **kwargs):
        dataset = tf.data.TFRecordDataset(predict_record_files, compression_type=self.record_option)
        dataset = dataset.repeat(self.predict_repeat_count)
        dataset = dataset.map(self._parse_example_fn)
        dataset = dataset.batch(
            self.predict_batch_size, drop_remainder=self.predict_drop_remainder
        ).prefetch(self.prefetch_size)
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
        relations = example['relations']
        return (input_ids, input_mask, segment_ids, predictions, relations)
