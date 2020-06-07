import tensorflow as tf

from transformers_keras.tokenizers import TransformerAbstractTokenizer

from .abstract_dataset_builder import AbstractDatasetBuilder


class TransformerDatasetBuilder(AbstractDatasetBuilder):

    def __init__(self, pad_id=0, **kwargs):
        super().__init__(**kwargs)
        self.pad_id = pad_id

    def _repeat(self, dataset, mode='train'):
        repeat = 0
        if 'train' == mode:
            repeat = self.train_repeat_count
        elif 'valid' == mode:
            repeat = self.valid_repeat_count
        elif 'predict' == mode:
            repeat = self.predict_repeat_count
        else:
            raise ValueError('Invalid mode: %s, must be in [train, valid, predict]' % mode)
        if not repeat or repeat <= 0:
            return dataset
        dataset = dataset.repeat(repeat)
        return dataset

    def _shuffle(self, dataset, mode='train'):
        if 'predict' == mode:
            return dataset
        elif 'train' == mode:
            dataset = dataset.shuffle(
                buffer_size=self.train_shuffle_buffer_size,
                seed=self.train_shuffle_seed,
                reshuffle_each_iteration=self.train_reshuffle_each_iteration)
            return dataset
        elif 'valid' == mode:
            dataset = dataset.shuffle(
                buffer_size=self.valid_shuffle_buffer_size,
                seed=self.valid_shuffle_seed,
                reshuffle_each_iteration=self.valid_reshuffle_each_iteration)
            return dataset
        else:
            raise ValueError('Invalid mode: %s, must be in [train, valid, predict]' % mode)

    def build_train_dataset(self, train_files, **kwargs):
        raise NotImplementedError()

    def build_valid_dataset(self, valid_files, **kwargs):
        raise NotImplementedError()

    def build_predict_dataset(self, valid_files, **kwargs):
        raise NotImplementedError()


class TransformerTFRecordDatasetBuilder(TransformerDatasetBuilder):
    """Build dataset from tfrecord files."""

    # TODO: Shift tgt sequence when generating tfrecord files

    def __init__(self, src_max_len=512, tgt_max_len=512, record_option='GZIP', **kwargs):
        super().__init__(**kwargs)
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        self.compression_type = record_option

    def _build_dataset_from_tfrecord_files(self, files, skip=0):
        dataset = tf.data.Dataset.from_tensor_slices(files)
        dataset = dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type=self.compression_type).skip(skip),
            cycle_length=len(files)
        )
        return dataset

    def _parse_example_fn(self, x):
        name_to_features = {
            'src_ids': tf.io.FixedLenFeature([self.src_max_len], tf.int64),
            'tgt_ids': tf.io.FixedLenFeature([self.tgt_max_len], tf.int64)
        }
        example = tf.io.parse_single_example(x, name_to_features)
        features = example['src_ids']
        labels = example['tgt_ids']
        return (features, labels)

    def build_train_dataset(self, train_files, **kwargs):
        dataset = self._build_dataset_from_tfrecord_files(train_files, skip=self.train_skip_count)
        dataset = self._repeat(dataset, mode='train')
        dataset = self._shuffle(dataset, mode='train')
        dataset = dataset.map(lambda x: self._parse_example_fn(x), num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.filter(lambda x, y: tf.logical_and(tf.size(x) > 0, tf.size(y) > 0))
        dataset = dataset.batch(self.train_batch_size, drop_remainder=self.train_drop_remainder)
        # should shift tgt sequence when generating tfrecord files
        dataset = dataset.map(lambda x, y: ((x, y[:, :-1]), y[:, 1:]), num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.prefetch(self.prefetch_size)
        return dataset

    def build_valid_dataset(self, valid_files, **kwargs):
        dataset = self._build_dataset_from_tfrecord_files(valid_files, skip=self.valid_skip_count)
        dataset = self._repeat(dataset, mode='valid')
        dataset = self._shuffle(dataset, mode='valid')
        dataset = dataset.map(lambda x: self._parse_example_fn(x), num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.filter(lambda x, y: tf.logical_and(tf.size(x) > 0, tf.size(y) > 0))
        dataset = dataset.batch(self.train_batch_size, drop_remainder=self.valid_drop_remainder)
        dataset = dataset.map(lambda x, y: ((x, y[:, :-1]), y[:, 1:]), num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.prefetch(self.prefetch_size)
        return dataset

    def build_predict_dataset(self, predict_files, **kwargs):
        # prediction process is different from training and evaluation
        raise NotImplementedError()


class TransformerTextFileDatasetBuilder(TransformerDatasetBuilder):

    def __init__(self,
                 src_tokenizer: TransformerAbstractTokenizer,
                 tgt_tokenizer: TransformerAbstractTokenizer,
                 pad_id=0,
                 **kwargs):
        super().__init__(pad_id=pad_id, **kwargs)
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.sep = kwargs.get('sep', '@@@')

    def _build_dataset_from_text_files(self, files, mode='train'):
        assert all(isinstance(x, tuple) for x in files), 'Invalid input files format.'
        src_files, tgt_files = [x[0] for x in files], [x[1] for x in files]
        src_dataset = tf.data.Dataset.from_tensor_slices(src_files)
        tgt_dataset = tf.data.Dataset.from_tensor_slices(tgt_files)

        skip = 0
        if 'train' == mode:
            skip = self.train_skip_count
        elif 'valid' == mode:
            skip = self.valid_skip_count
        elif 'predict' == mode:
            skip = self.predict_skip_count
        else:
            skip = 0
        if not skip or skip <= 0:
            skip = 0

        src_dataset = src_dataset.flat_map(lambda x: tf.data.TextLineDataset(x).skip(skip))
        tgt_dataset = tgt_dataset.flat_map(lambda x: tf.data.TextLineDataset(x).skip(skip))
        dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
        return dataset

    def _parse_example(self, dataset):

        def _parse(x, y):
            src = x.numpy().decode('utf8')
            tgt = y.numpy().decode('utf8')
            src = [self.src_tokenizer.sos_id] + self.src_tokenizer.encode(src) + [self.src_tokenizer.eos_id]
            tgt = [self.tgt_tokenizer.sos_id] + self.tgt_tokenizer.encode(tgt) + [self.tgt_tokenizer.eos_id]
            return src, tgt

        def _tf_parse_fn(x, y):
            src, tgt = tf.py_function(_parse, [x, y], [tf.int64, tf.int64])
            src.set_shape([None])
            tgt.set_shape([None])
            return src, tgt

        dataset = dataset.map(lambda x, y: _tf_parse_fn(x, y), num_parallel_calls=self.num_parallel_calls)
        return dataset

    def build_train_dataset(self, train_files, **kwargs):
        dataset = self._build_dataset_from_text_files(train_files, mode='train')
        dataset = self._repeat(dataset, mode='train')
        dataset = self._shuffle(dataset, mode='train')
        dataset = self._parse_example(dataset)
        dataset = dataset.filter(lambda x, y: tf.logical_and(tf.size(x) > 0, tf.size(y) > 0))

        pad_id = tf.constant(self.pad_id, dtype=tf.int64)
        dataset = dataset.padded_batch(
            batch_size=self.train_batch_size,
            padded_shapes=([None], [None]),
            padding_values=(pad_id, pad_id),
            drop_remainder=self.train_drop_remainder,
        ).prefetch(self.prefetch_size)
        # x-> encoder input, y->decoder input, z->label
        dataset = dataset.map(lambda x, y: ((x, y[:, :-1]), y[:, 1:]))
        return dataset

    def build_valid_dataset(self, valid_files, **kwargs):
        dataset = self._build_dataset_from_text_files(valid_files, mode='valid')
        dataset = self._repeat(dataset, mode='valid')
        dataset = self._shuffle(dataset, mode='valid')
        dataset = self._parse_example(dataset)
        dataset = dataset.filter(lambda x, y: tf.logical_and(tf.size(x) > 0, tf.size(y) > 0))

        pad_id = tf.constant(self.pad_id, dtype=tf.int64)
        dataset = dataset.padded_batch(
            batch_size=self.valid_batch_size,
            padded_shapes=([None], [None]),
            padding_values=(pad_id, pad_id),
            drop_remainder=self.valid_drop_remainder,
        ).prefetch(self.prefetch_size)
        # x-> encoder input, y->decoder input, z->label
        dataset = dataset.map(lambda x, y: ((x, y[:, :-1]), y[:, 1:]))
        return dataset

    def build_predict_dataset(self, predict_files, **kwargs):
        # prediction process is different from training and evaluation
        raise NotImplementedError()
