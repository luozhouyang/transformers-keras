import tensorflow as tf

from transformers_keras.tokenizers.abstract_tokenizer import AbstractTokenizer


class Dataset:
    """Build data input pipeline using `tf.data.Dataset` for models."""

    def __init__(self, src_tokenizer: AbstractTokenizer, tgt_tokenizer: AbstractTokenizer, config: dict = None):
        """Constructor.

        Args:
            src_tokenizer: An instance of Tokenizer, used to convert src language's tokens to ids
            tgt_tokenizer: An instance of Tokenizer, used to convert tgt language's tokens to ids
        """
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        default_config = self._get_default_config()
        if config:
            default_config.update(config)
        self.config = default_config

    def build_train_dataset(self, train_files):
        """Build dataset for training.

        Args:
            train_files: An iterable of tuple (src_file, tgt_file)

        Returns:
            A tf.data.Dataset object
        """
        raise NotImplementedError()

    def build_eval_dataset(self, eval_files):
        """Build dataset for evaluation.

        Args:
            eval_files: An iterable of tuple (src_file, tgt_file)

        Returns:
            A tf.data.Dataset object
        """
        raise NotImplementedError()

    def build_predict_dataset(self, predict_files):
        """Build dataset for prediction.

        Args:
            predict_files:  An iterable of src_file, no tgt_file for prediction

        Returns:
            A tf.data.Dataset object
        """
        raise NotImplementedError()

    def _shuffle(self, dataset):
        """Shuffle dataset."""
        dataset = dataset.shuffle(
            buffer_size=self.config['buffer_size'],
            seed=self.config['seed'],
            reshuffle_each_iteration=self.config['reshuffle_each_iteration'])
        return dataset

    def _split_line(self, dataset):
        """Split line to string tokens."""
        dataset = dataset.map(
            lambda x, y: (
                tf.strings.split([x], sep=self.config['sequence_sep']).values,
                tf.strings.split([y], sep=self.config['sequence_sep']).values,),
            num_parallel_calls=self.config['num_parallel_calls']
        ).prefetch(self.config['prefetch_size'])
        return dataset

    def _split_line_for_predict(self, dataset):
        """Split line to string tokens for predict mode."""
        dataset = dataset.map(
            lambda x: (
                tf.strings.split([x], sep=self.config['sequence_sep']).values),
            num_parallel_calls=self.config['num_parallel_calls']
        ).prefetch(self.config['prefetch_size'])
        return dataset

    def _filter_dataset(self, dataset: tf.data.Dataset):
        """Filter examples which are empty or too long."""
        # filter empty sequences
        dataset = dataset.filter(lambda x, y: tf.logical_and(tf.size(x) > 0, tf.size(y) > 0))
        # length filter
        x_max_len = self.config.get('x_max_len', -1)
        if x_max_len > 0:
            dataset = dataset.filter(lambda x, y: tf.size(x) <= x_max_len)
        y_max_len = self.config.get('y_max_len', -1)
        if y_max_len > 0:
            dataset = dataset.filter(lambda x, y: tf.size(y) <= y_max_len)
        return dataset

    def _filter_dataset_for_predict(self, dataset):
        """Filter examples which are empty or too long for predict mode."""
        # filter empty sequences
        dataset = dataset.filter(lambda x: tf.size(x) > 0)
        # length filter
        x_max_len = self.config.get('x_max_len', -1)
        if x_max_len > 0:
            dataset = dataset.filter(lambda x: tf.size(x) <= x_max_len)
        return dataset

    def _convert_tokens_to_ids(self, dataset):
        """Convert string tokens to ids."""
        dataset = dataset.map(
            lambda x, y: (
                self.src_tokenizer.encode(x),
                self.tgt_tokenizer.encode(y)),
            num_parallel_calls=self.config['num_parallel_calls']
        ).prefetch(self.config['prefetch_size'])
        return dataset

    def _convert_tokens_to_ids_for_predict(self, dataset):
        """Convert string tokens to ids for predict mode."""
        dataset = dataset.map(
            lambda x: self.src_tokenizer.encode(x),
            num_parallel_calls=self.config['num_parallel_calls']
        ).prefetch(self.config['prefetch_size'])
        return dataset

    def _add_sos_and_eos(self, dataset):
        """Add sos in the start and eos in the end."""
        src_sos_id = tf.constant(self.src_tokenizer.sos_id, dtype=tf.dtypes.int64)
        src_eos_id = tf.constant(self.src_tokenizer.eos_id, dtype=tf.dtypes.int64)
        tgt_sos_id = tf.constant(self.tgt_tokenizer.sos_id, dtype=tf.dtypes.int64)
        tgt_eos_id = tf.constant(self.tgt_tokenizer.eos_id, dtype=tf.dtypes.int64)
        if self.config['add_sos']:
            dataset = dataset.map(
                lambda x, y: (tf.concat(([src_sos_id], x), axis=0), tf.concat(([tgt_sos_id], y), axis=0)),
                num_parallel_calls=self.config['num_parallel_calls']
            ).prefetch(self.config['prefetch_size'])
        if self.config['add_sos']:
            dataset = dataset.map(
                lambda x, y: (tf.concat((x, [src_eos_id]), axis=0), tf.concat((y, [tgt_eos_id]), axis=0)),
                num_parallel_calls=self.config['num_parallel_calls']
            ).prefetch(self.config['prefetch_size'])
        return dataset

    def _add_sos_and_eos_for_predict(self, dataset):
        """Add sos in the start and eos in the end for predict mode."""
        src_sos_id = tf.constant(self.src_tokenizer.sos_id, dtype=tf.dtypes.int64)
        src_eos_id = tf.constant(self.src_tokenizer.eos_id, dtype=tf.dtypes.int64)
        if self.config['add_sos']:
            dataset = dataset.map(
                lambda x: tf.concat(([src_sos_id], x), axis=0),
                num_parallel_calls=self.config['num_parallel_calls']
            ).prefetch(self.config['prefetch_size'])
        if self.config['add_sos']:
            dataset = dataset.map(
                lambda x: tf.concat((x, [src_eos_id]), axis=0),
                num_parallel_calls=self.config['num_parallel_calls']
            ).prefetch(self.config['prefetch_size'])
        return dataset

    def _padding_and_batching(self, dataset, batch_size, padding_value):
        x_padded_shape = self.config['x_max_len'] if self.config['x_max_len'] > 0 else None
        y_padded_shape = self.config['y_max_len'] if self.config['y_max_len'] > 0 else None
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padding_values=(padding_value, padding_value),
            padded_shapes=([x_padded_shape], [y_padded_shape]))
        return dataset

    def _padding_and_batching_for_predict(self, dataset, batch_size, padding_value):
        x_padded_shape = self.config['x_max_len'] if self.config['x_max_len'] > 0 else None
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padding_values=padding_value,
            padded_shapes=[x_padded_shape])
        return dataset

    def _build_dataset_from_files(self, files):
        """Build `tf.data.Dataset` object from files for training or evaluation.

        Args:
            files: An iterable of tuple (src_file, tgt_file)

        Returns:
            A tf.data.Dataset object
        """
        src_files, tgt_files = files
        src_dataset = tf.data.Dataset.from_tensor_slices(src_files)
        src_dataset = src_dataset.flat_map(lambda x: tf.data.TextLineDataset(x).skip(self.config.get('skip_count', 0)))
        tgt_dataset = tf.data.Dataset.from_tensor_slices(tgt_files)
        tgt_dataset = tgt_dataset.flat_map(lambda x: tf.data.TextLineDataset(x).skip(self.config.get('skip_count', 0)))
        dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
        return dataset

    def _build_dataset_from_files_for_predict(self, files):
        """Build `tf.data.Dataset` object from files for prediction.

        Args:
            files: An iterable of src_file, no tgt_file for prediction.

        Returns:
            A tf.data.Dataset object
        """
        src_files = files
        src_dataset = tf.data.Dataset.from_tensor_slices(src_files)
        src_dataset = src_dataset.flat_map(lambda x: tf.data.TextLineDataset(x).skip(self.config.get('skip_count', 0)))
        return src_dataset

    @staticmethod
    def _get_default_config():
        c = {
            'skip_count': 0,
            'num_parallel_calls': tf.data.experimental.AUTOTUNE,
            'prefetch_size': tf.data.experimental.AUTOTUNE,
            'buffer_size': 1000000,  # must be greater than num of examples
            'seed': None,
            'reshuffle_each_iteration': True,
            'add_sos': True,
            'add_eos': True,
            'x_max_len': -1,
            'y_max_len': -1,
            'train_batch_size': 32,
            'eval_batch_size': 32,
            'predict_batch_size': 32,
            'sequence_sep': ' ',
        }
        return c
