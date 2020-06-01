import tensorflow as tf

from transformers_keras.tokenizers import TransformerAbstractTokenizer

from .abstract_dataset_builder import AbstractDatasetBuilder


class TransformerDatasetBuilder(AbstractDatasetBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pad_id = kwargs.get('pad_id', 0)

    def _build_dataset(
            self,
            files,
            skip=0,
            repeat=0,
            buffer_size=-1,
            seed=None,
            reshuffle=True,
            parallels=tf.data.experimental.AUTOTUNE,
            pad_id=0,
            prefetch_size=tf.data.experimental.AUTOTUNE,
            batch_size=32,
            drop_remainder=False):
        dataset = self._create_dataset_from_files(files, skip)
        dataset = dataset.repeat(repeat)
        dataset = dataset.shuffle(
            buffer_size=buffer_size,
            seed=seed,
            reshuffle_each_iteration=reshuffle)
        dataset = dataset.map(self._parse_example_fn, num_parallel_calls=parallels)
        dataset = dataset.filter(lambda x, y: tf.logical_and(tf.size(x) > 0, tf.size(y) > 0))
        pad_id = tf.constant(pad_id, dtype=tf.int64)
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=([None], [None]),
            padding_values=(pad_id, pad_id),
            drop_remainder=drop_remainder,
        ).prefetch(prefetch_size)
        # x-> encoder input, y->decoder input, z->label
        dataset = dataset.map(lambda x, y: (x, y[:, :-1], y[:, 1:]))
        return dataset

    def build_train_dataset(self, train_files):
        dataset = self._build_dataset(
            files=train_files,
            skip=self.train_skip_count,
            repeat=self.train_repeat_count,
            buffer_size=self.train_shuffle_buffer_size,
            seed=self.train_shuffle_seed,
            reshuffle=self.train_reshuffle_each_iteration,
            paralles=self.num_parallel_calls,
            pad_id=self.pad_id,
            prefetch_size=self.prefetch_size,
            batch_size=self.train_batch_size,
            drop_remainder=self.train_drop_remainder
        )
        dataset = dataset.map(lambda x, y, z: ((x, y), z))
        return dataset

    def build_valid_dataset(self, valid_files):
        dataset = self._build_dataset(
            files=valid_files,
            skip=self.valid_skip_count,
            repeat=self.valid_repeat_count,
            buffer_size=self.valid_shuffle_buffer_size,
            seed=self.valid_shuffle_seed,
            reshuffle=self.valid_reshuffle_each_iteration,
            parallels=self.num_parallel_calls,
            pad_id=self.pad_id,
            prefetch_size=self.prefetch_size,
            batch_size=self.valid_batch_size,
            drop_remainder=self.valid_drop_remainder
        )
        dataset = dataset.map(lambda x, y, z: ((x, y), z))
        return dataset

    def build_predict_dataset(self, predict_files):
        dataset = self._build_dataset(
            files=predict_files,
            skip=self.predict_skip_count,
            repeat=self.predict_repeat_count,
            buffer_size=self.predict_shuffle_buffer_size,
            seed=self.predict_shuffle_seed,
            reshuffle=self.predict_reshuffle_each_iteration,
            parallels=self.num_parallel_calls,
            pad_id=self.pad_id,
            prefetch_size=self.prefetch_size,
            batch_size=self.predict_batch_size,
            drop_remainder=self.predict_drop_remainder
        )
        dataset = dataset.map(lambda x, y, z: (x, y))
        return dataset

    def _create_dataset_from_files(self, files, skip=0):
        raise NotImplementedError()

    def _parse_example_fn(self, x):
        raise NotImplementedError()


class TransformerTFRecordDatasetBuilder(TransformerDatasetBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.src_max_len = kwargs.get('src_max_len', 16)
        self.tgt_max_len = kwargs.get('tgt_max_len', 16)

    def _create_dataset_from_files(self, files, skip=0):
        dataset = tf.data.Dataset.from_tensor_slices(files)
        dataset = dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x).skip(skip),
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


class TransformerTextFileDatasetBuilder(TransformerDatasetBuilder):

    def __init__(self, tokenizer: TransformerAbstractTokenizer, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.sep = kwargs.get('sep', '@@@')
        self.pad_id = self.tokenizer.pad_id

    def _create_dataset_from_files(self, files, skip=0):
        """Create tf.data.Dataset instance from files.

        Args:
            files: Python list, can be:
                1) A list of files, each line of each file contains src and tgt sequence.
                2) A list of (src_file, tgt_file) tuples
            skip: Python integer, skip count of the file

        Returns:
            An tf.data.Dataset instance
        """

        if not files:
            raise ValueError('Invalid argument `files`.')

        ele = files[0]

        if isinstance(ele, str):
            dataset = tf.data.Dataset.from_tensor_slices(files)
            dataset = dataset.interleave(
                lambda x: tf.data.TextLineDataset(x).skip(skip),
                cycle_length=len(files)
            )
            return dataset
        elif isinstance(ele, tuple):
            src_files = [x[0] for x in files]
            tgt_files = [x[1] for x in files]
            src_dataset = tf.data.Dataset.from_tensor_slices(src_files)
            tgt_dataset = tf.data.Dataset.from_tensor_slices(tgt_files)
            src_dataset = src_dataset.flat_map(lambda x: tf.data.TextLineDataset(x).skip(skip))
            tgt_dataset = tgt_dataset.flat_map(lambda x: tf.data.TextLineDataset(x).skip(skip))
            dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
            return dataset

        else:
            raise ValueError('Invalid argument `files`')

    def _parse_example_fn(self, x):

        def _parse(record):
            pair = record.numpy().split(self.sep)
            a, b = pair[0], pair[1]
            src = [self.tokenizer.sos_id] + self.encode(a) + [self.tokenizer.eos_id]
            tgt = [self.tokenizer.sos_id] + self.encode(b) + [self.tokenizer.eos_id]
            return src, tgt

        src, tgt = tf.py_function(_parse, [x], [tf.int64, tf.int64])
        src.set_shape([None])
        tgt.set_shape([None])
        return src, tgt
