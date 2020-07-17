import abc

import tensorflow as tf


class AbstractDatasetBuilder(abc.ABC):

    def __init__(self, **kwargs):
        super().__init__()

    def build_train_dataset(self, train_files, **kwargs):
        raise NotImplementedError()

    def build_valid_dataset(self, valid_files, **kwargs):
        raise NotImplementedError()

    def build_predict_dataset(self, valid_files, **kwargs):
        raise NotImplementedError()
