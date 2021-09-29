import abc
import json
import logging
import os

import numpy as np
import tensorflow as tf


def zip_weights(model, ckpt, variables_mapping, self_weight_names, **kwargs):
    weights, values = [], []
    used_weights = [w for w in model.trainable_weights if w.name in self_weight_names]
    for w in used_weights:
        var = variables_mapping.get(w.name, None)
        if var is None:
            logging.warning("Model weight: %s not collected in weights mapping.", w.name)
            continue
        v = tf.train.load_variable(ckpt, var)
        if w.name == "bert/nsp/dense/kernel:0":
            v = v.T
        weights.append(w)
        values.append(v)
        if kwargs.get("verbose", True):
            logging.info("Load weight: {:60s} <-- {}".format(w.name, variables_mapping[w.name]))

    mapped_values = zip(weights, values)
    return mapped_values


def parse_pretrained_model_files(pretrained_model_dir):
    config_file, ckpt, vocab = None, None, None
    pretrained_model_dir = os.path.abspath(pretrained_model_dir)
    if not os.path.exists(pretrained_model_dir):
        logging.info("pretrain model dir: {} is not exists.".format(pretrained_model_dir))
        return config_file, ckpt, vocab
    for f in os.listdir(pretrained_model_dir):
        if "config" in str(f) and str(f).endswith(".json"):
            config_file = os.path.join(pretrained_model_dir, f)
        if "vocab" in str(f):
            vocab = os.path.join(pretrained_model_dir, f)
        if "ckpt" in str(f):
            n = ".".join(str(f).split(".")[:-1])
            ckpt = os.path.join(pretrained_model_dir, n)
    return config_file, ckpt, vocab


class AbstractAdapter(abc.ABC):
    """Abstract model weights adapter."""

    def __init__(
        self,
        skip_token_embedding=False,
        skip_position_embedding=False,
        skip_segment_embedding=False,
        skip_embedding_layernorm=False,
        skip_pooler=False,
        skip_mlm=True,
        **kwargs
    ):
        super().__init__()
        self.skip_token_embedding = skip_token_embedding
        self.skip_position_embedding = skip_position_embedding
        self.skip_segment_embedding = skip_segment_embedding
        self.skip_embedding_layernorm = skip_embedding_layernorm
        self.skip_pooler = skip_pooler
        self.skip_mlm = skip_mlm

        logging.info(
            "Adapter skipping config: %s",
            json.dumps(
                {
                    "skip_token_embedding": self.skip_token_embedding,
                    "skip_position_embedding": self.skip_position_embedding,
                    "skip_segment_embedding": self.skip_segment_embedding,
                    "skip_embedding_layernorm": self.skip_embedding_layernorm,
                    "skip_pooler": self.skip_pooler,
                    "skip_mlm": self.skip_mlm,
                }
            ),
        )

    def parse_files(self, model_path, **kwargs):
        config_file, ckpt, vocab = parse_pretrained_model_files(model_path)
        return {
            "config_file": config_file,
            "ckpt": ckpt,
            "vocab_file": vocab,
        }

    @abc.abstractmethod
    def adapte_config(self, config_file, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def adapte_weights(self, model, ckpt, model_config, use_functional_api=True, **kwargs):
        raise NotImplementedError()

    def _skip_weight(self, mapping, name):
        mapping.pop(name)
        logging.info("Skip load weight: %s", name)

    def _check_weights(self, mapping, model, ckpt, **kwargs):
        self_weights = {w.name: w.numpy() for w in model.trainable_weights}
        for k, v in self_weights.items():
            ckpt_weight = mapping.get(k, None)
            if not ckpt_weight:
                continue
            ckpt_value = tf.train.load_variable(ckpt, ckpt_weight)
            if ckpt_value is None:
                logging.warning("ckpt value is None of key: %s", ckpt_weight)
            assert np.allclose(v, ckpt_value)
        logging.info("All weights value are checked.")
