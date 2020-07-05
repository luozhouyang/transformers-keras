import abc
import logging
import os

import tensorflow as tf


class AbstractStrategy(abc.ABC):
    """Pretrained model load strategy."""

    def mapping_config(self, pretrained_config_file):
        """Convert pretrained configs to model configs.

        Args:
            pretrained_config_file: File path of pretrained model's config

        Returns:
            A python dict, model config
        """
        raise NotImplementedError()

    def build_model(self, model_config):
        """Build and compile model accroding to model config.

        Args:
            model_config: A python dict, model's config

        Returns:
            A keras model, instance of `tf.keras.Model`, compiled.
        """
        raise NotImplementedError()

    def mapping_variables(self, model_config, model, ckpt):
        """Mapping pretrained variables to model's variables.

        Args:
            model_config: A python dict, model's config
            model: A keras model, compiled
            ckpt: Python str, pretrained checkpoint model.

        Returns:
            Python dict, variables' name mapping
        """
        raise NotImplementedError()

    def zip_weights(self, model, ckpt, variables_mapping):
        """Zip weights and values.

        Args:
            model: A keras model, compiled
            ckpt: Python str, pretrained checkpoint model
            variables_mapping: Python dict, variables' name mapping

        Returns:
            A List of tuple (model_weight, pretrained_weight)
        """
        raise NotImplementedError()


class AbstractAdapter(abc.ABC):

    @abc.abstractmethod
    def adapte(self, pretrain_model_dir, **kwargs):
        raise NotImplementedError()

    def _parse_files(self, pretrain_model_dir):
        config_file, ckpt, vocab = None, None, None
        if not os.path.exists(pretrain_model_dir):
            logging.info('pretrain model dir: {} is not exists.'.format(pretrain_model_dir))
            return
        for f in os.listdir(pretrain_model_dir):
            if str(f).endswith('config.json'):
                config_file = os.path.join(pretrain_model_dir, f)
            if 'vocab' in str(f):
                vocab = os.path.join(pretrain_model_dir, f)
            if 'ckpt' in str(f):
                n = '.'.join(str(f).split('.')[:-1])
                ckpt = os.path.join(pretrain_model_dir, n)
        return config_file, ckpt, vocab


class PretrainedModelAdapter(AbstractAdapter):
    """Base class of pretrain models' adapter."""

    def __init__(self, strategy: AbstractStrategy):
        """Init.

        Args:
            strategy: An instance of `AbstractStrategy`
        """
        super().__init__()
        self.strategy = strategy

    def adapte(self, pretrain_model_dir, **kwargs):
        config_file, ckpt, vocab_file = self._parse_files(pretrain_model_dir)
        model_config = self.strategy.mapping_config(config_file)
        model = self.strategy.build_model(model_config)
        names_mapping = self.strategy.mapping_variables(model_config, model, ckpt)
        weights_values = self.strategy.zip_weights(model, ckpt, names_mapping)
        tf.keras.backend.batch_set_value(weights_values)
        return model, vocab_file
