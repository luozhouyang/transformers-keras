import abc

import tensorflow as tf


class AbstractDatasetBuilder(abc.ABC):

    def __init__(self, **kwargs):
        super().__init__()
        self.num_parallel_calls = kwargs.get('num_parallel_calls', tf.data.experimental.AUTOTUNE)
        self.prefetch_size = kwargs.get('prefetch_size', tf.data.experimental.AUTOTUNE)

        self.train_batch_size = kwargs.get('train_batch_size', 32)
        self.train_skip_count = kwargs.get('train_skip_count', 0)
        self.train_repeat_count = kwargs.get('train_repeat_count', None)
        self.train_shuffle_buffer_size = kwargs.get('train_shuffle_buffer_size', 1000000)
        self.train_shuffle_seed = kwargs.get('train_shuffle_seed', None)
        self.train_reshuffle_each_iteration = kwargs.get('train_reshuffle_each_iteration', True)
        self.train_drop_remainder = kwargs.get('train_drop_remainder', False)

        self.valid_batch_size = kwargs.get('valid_batch_size', 32)
        self.valid_skip_count = kwargs.get('valid_skip_count', 0)
        self.valid_repeat_count = kwargs.get('valid_repeat_count', None)
        self.valid_shuffle_buffer_size = kwargs.get('valid_shuffle_buffer_size', -1)
        self.valid_shuffle_seed = kwargs.get('valid_shuffle_seed', None)
        self.valid_reshuffle_each_iteration = kwargs.get('valid_reshuffle_each_iteration', True)
        self.valid_drop_remainder = kwargs.get('valid_drop_remainder', False)

        self.predict_batch_size = kwargs.get('predict_batch_size', 32)
        self.predict_skip_count = kwargs.get('predict_skip_count', 0)
        self.predict_repeat_count = kwargs.get('predict_repeat_count', None)
        self.predict_shuffle_buffer_size = kwargs.get('predict_shuffle_buffer_size', -1)
        self.predict_shuffle_seed = kwargs.get('predict_shuffle_seed', None)
        self.predict_reshuffle_each_iteration = kwargs.get('predict_reshuffle_each_iteration', True)
        self.predict_drop_remainder = kwargs.get('predict_drop_remainder', False)

    def build_train_dataset(self, train_files, **kwargs):
        raise NotImplementedError()

    def build_valid_dataset(self, valid_files, **kwargs):
        raise NotImplementedError()

    def build_predict_dataset(self, valid_files, **kwargs):
        raise NotImplementedError()
