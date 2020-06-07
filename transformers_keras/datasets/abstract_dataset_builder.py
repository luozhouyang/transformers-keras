import abc

import tensorflow as tf


class AbstractDatasetBuilder(abc.ABC):

    def __init__(self,
                 train_batch_size=32,
                 train_skip_count=0,
                 train_repeat_count=None,
                 train_shuffle_buffer_size=1000000,
                 train_shuffle_seed=None,
                 train_reshuffle_each_iteration=True,
                 train_drop_remainder=False,
                 valid_batch_size=32,
                 valid_skip_count=0,
                 valid_repeat_count=None,
                 valid_shuffle_buffer_size=100000,
                 valid_shuffle_seed=None,
                 valid_reshuffle_each_iteration=True,
                 valid_drop_remainder=False,
                 predict_batch_size=1,
                 predict_skip_count=0,
                 predict_repeat_count=None,
                 predict_drop_remainder=False,
                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
                 prefetch_size=tf.data.experimental.AUTOTUNE,
                 **kwargs):
        super().__init__()
        self.num_parallel_calls = num_parallel_calls
        self.prefetch_size = prefetch_size

        self.train_batch_size = train_batch_size
        self.train_skip_count = train_skip_count
        self.train_repeat_count = train_repeat_count
        self.train_shuffle_buffer_size = train_shuffle_buffer_size
        self.train_shuffle_seed = train_shuffle_seed
        self.train_reshuffle_each_iteration = train_reshuffle_each_iteration
        self.train_drop_remainder = train_drop_remainder

        self.valid_batch_size = valid_batch_size
        self.valid_skip_count = valid_skip_count
        self.valid_repeat_count = valid_repeat_count
        self.valid_shuffle_buffer_size = valid_shuffle_buffer_size
        self.valid_shuffle_seed = valid_shuffle_seed
        self.valid_reshuffle_each_iteration = valid_reshuffle_each_iteration
        self.valid_drop_remainder = valid_drop_remainder

        self.predict_batch_size = predict_batch_size
        self.predict_skip_count = predict_skip_count
        self.predict_repeat_count = predict_repeat_count
        self.predict_drop_remainder = predict_drop_remainder

    def build_train_dataset(self, train_files, **kwargs):
        raise NotImplementedError()

    def build_valid_dataset(self, valid_files, **kwargs):
        raise NotImplementedError()

    def build_predict_dataset(self, valid_files, **kwargs):
        raise NotImplementedError()
