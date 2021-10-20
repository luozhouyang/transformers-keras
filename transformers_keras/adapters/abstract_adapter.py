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

    @abc.abstractmethod
    def adapte_config(self, model_path, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def adapte_weights(self, model, model_config, model_path, **kwargs):
        raise NotImplementedError()


class AbstractBertAdapter(AbstractAdapter):
    """Abstract Bert adapter"""

    def __init__(
        self,
        use_functional_api=True,
        with_mlm=False,
        with_nsp=False,
        with_sop=False,
        skip_token_embedding=False,
        skip_position_embedding=False,
        skip_segment_embedding=False,
        skip_embedding_layernorm=False,
        skip_pooler=False,
        check_weights=True,
        verbose=True,
        **kwargs
    ):
        self.use_functional_api = use_functional_api
        self.with_mlm = with_mlm
        self.with_nsp = with_nsp
        self.with_sop = with_sop
        self.check_weights = check_weights
        self.verbose = verbose

        self.model_files = None
        self._pretrained_weights_map = {}

        # skip weights
        self.skip_token_embedding = skip_token_embedding
        self.skip_position_embedding = skip_position_embedding
        self.skip_segment_embedding = skip_segment_embedding
        self.skip_embedding_layernorm = skip_embedding_layernorm
        self.skip_pooler = skip_pooler
        self.weights_to_skip = set()
        logging.info(
            "Adapter skipping config: %s",
            json.dumps(
                {
                    "skip_token_embedding": self.skip_token_embedding,
                    "skip_position_embedding": self.skip_position_embedding,
                    "skip_segment_embedding": self.skip_segment_embedding,
                    "skip_embedding_layernorm": self.skip_embedding_layernorm,
                    "skip_pooler": self.skip_pooler,
                }
            ),
        )

    def _parse_files(self, model_path, **kwargs):
        config_file, ckpt, vocab = parse_pretrained_model_files(model_path)
        return {
            "config_file": config_file,
            "ckpt": ckpt,
            "vocab_file": vocab,
        }

    def adapte_config(self, model_path, **kwargs):
        if self.model_files is None:
            self.model_files = self._parse_files(model_path, **kwargs)
        config_file = self.model_files["config_file"]
        with open(config_file, mode="rt", encoding="utf8") as fin:
            config = json.load(fin)

        model_config = {
            "vocab_size": config["vocab_size"],
            "activation": config["hidden_act"],
            "max_positions": config["max_position_embeddings"],
            "hidden_size": config["hidden_size"],
            "type_vocab_size": config["type_vocab_size"],
            "intermediate_size": config["intermediate_size"],
            "hidden_dropout_rate": config["hidden_dropout_prob"],
            "attention_dropout_rate": config["attention_probs_dropout_prob"],
            "initializer_range": config["initializer_range"],
            "num_layers": config["num_hidden_layers"],
            "num_attention_heads": config["num_attention_heads"],
        }
        return model_config

    def adapte_weights(self, model, model_config, model_path, **kwargs):
        self._pretrained_weights_map = self._read_pretrained_weights(model_path, **kwargs)

        weights_mapping = {}
        bert_weights_mapping = self._adapte_bert_weights(model, model_config, **kwargs)
        weights_mapping.update(bert_weights_mapping)

        if self.with_mlm:
            mlm_weights = self._adapte_mlm_weights(model, model_config, **kwargs)
            weights_mapping.update(mlm_weights)

        if self.with_nsp:
            nsp_weights = self._adapte_nsp_weights(model, model_config, **kwargs)
            weights_mapping.update(nsp_weights)

        if self.with_sop:
            sop_weights = self._adapte_sop_weights(model, model_config, **kwargs)
            weights_mapping.update(sop_weights)

        # skip weights
        self._skipping_weights(model, **kwargs)

        take_values = set(weights_mapping.values())
        for k in self._pretrained_weights_map.keys():
            if k not in take_values:
                logging.info("pretrained weight: {} not used.".format(k))

        zipping_weights, zipping_values = self._zip_weights(model, model_config, weights_mapping, **kwargs)
        tf.keras.backend.batch_set_value(zip(zipping_weights, zipping_values))

        # check weights
        self._check_weights(model, zipping_weights, zipping_values, weights_mapping, **kwargs)

    @abc.abstractmethod
    def _read_pretrained_weights(self, model_path, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _adapte_bert_weights(self, model, model_config, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _adapte_mlm_weights(self, model, model_config, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _adapte_nsp_weights(self, model, model_config, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _adapte_sop_weights(self, model, model_config, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _zip_weights(self, model, model_config, weights_mapping, **kwargs):
        raise NotImplementedError()

    def _skipping_weights(self, model, **kwargs):
        model_prefix = model.bert_model.name if self.use_functional_api else model.name + "/" + model.bert_model.name

        def _skip(w):
            self.weights_to_skip.add(w)
            logging.info("Weights will be skipped to load: %s", w)

        if self.skip_token_embedding:
            _skip("{}/embeddings/word_embeddings:0".format(model_prefix))
        if self.skip_position_embedding:
            _skip("{}/embeddings/position_embeddings:0".format(model_prefix))
        if self.skip_segment_embedding:
            _skip("{}/embeddings/token_type_embeddings:0".format(model_prefix))
        if self.skip_embedding_layernorm:
            _skip("{}/embeddings/LayerNorm/gamma:0".format(model_prefix))
            _skip("{}/embeddings/LayerNorm/beta:0".format(model_prefix))
        if self.skip_pooler:
            _skip("{}/pooler/dense/kernel:0".format(model_prefix))
            _skip("{}/pooler/dense/bias:0".format(model_prefix))

    def _check_weights(self, model, zipping_weights, zipping_values, weights_mapping, **kwargs):
        if not self.check_weights:
            logging.info("Skipped to check weights due to option `check_weights` set to `False`")
            return
        for k, v in zip(zipping_weights, zipping_values):
            vv = self._pretrained_weights_map[weights_mapping[k.name]]
            try:
                assert np.allclose(v, vv)
            except Exception as e:
                logging.warning("{} & {} not close!".format(k, weights_mapping[k.name]))
                logging.warning("{} -> \n {}".format(k, v))
                logging.warning("{} -> \n {}".format(weights_mapping[k.name], vv))
                logging.warning(e)
                logging.warning("=" * 80)


class AbstractAlbertAdapter(AbstractAdapter):
    """Abstract adapter for albert"""

    def __init__(
        self,
        use_functional_api=True,
        with_mlm=False,
        with_nsp=False,
        with_sop=False,
        skip_embedding_mapping_in=False,
        skip_token_embedding=False,
        skip_position_embedding=False,
        skip_segment_embedding=False,
        skip_embedding_layernorm=False,
        skip_pooler=False,
        verbose=True,
        check_weights=True,
        **kwargs
    ) -> None:
        self.use_functional_api = use_functional_api
        self.with_mlm = with_mlm
        self.with_nsp = with_nsp
        self.with_sop = with_sop
        self.check_weights = check_weights
        self.verbose = verbose

        # model files
        self.model_files = None

        # skip weights
        self.skip_embedding_mapping_in = skip_embedding_mapping_in
        self.skip_token_embedding = skip_token_embedding
        self.skip_position_embedding = skip_position_embedding
        self.skip_segment_embedding = skip_segment_embedding
        self.skip_embedding_layernorm = skip_embedding_layernorm
        self.skip_pooler = skip_pooler
        logging.info(
            "Adapter skipping config: %s",
            json.dumps(
                {
                    "skip_token_embedding": self.skip_token_embedding,
                    "skip_position_embedding": self.skip_position_embedding,
                    "skip_segment_embedding": self.skip_segment_embedding,
                    "skip_embedding_layernorm": self.skip_embedding_layernorm,
                    "skip_pooler": self.skip_pooler,
                    "skip_embedding_mapping_in": self.skip_embedding_mapping_in,
                }
            ),
        )

    def _parse_files(self, model_path, **kwargs):
        config_file, ckpt, vocab = parse_pretrained_model_files(model_path)
        return {
            "config_file": config_file,
            "ckpt": ckpt,
            "vocab_file": vocab,
        }

    def adapte_config(self, model_path, **kwargs):
        if self.model_files is None:
            self.model_files = self._parse_files(model_path, **kwargs)
        config_file = self.model_files["config_file"]
        with open(config_file, mode="rt", encoding="utf8") as fin:
            config = json.load(fin)

        model_config = {
            "vocab_size": config["vocab_size"],
            "max_positions": config["max_position_embeddings"],
            "embedding_size": config["embedding_size"],
            "type_vocab_size": config["type_vocab_size"],
            "num_layers": config["num_hidden_layers"],
            "num_groups": config["num_hidden_groups"],
            "num_layers_each_group": config["inner_group_num"],
            "hidden_size": config["hidden_size"],
            "num_attention_heads": config["num_attention_heads"],
            "intermediate_size": config["intermediate_size"],
            "activation": config["hidden_act"],
            "hidden_dropout_rate": config["hidden_dropout_prob"],
            "attention_dropout_rate": config["attention_probs_dropout_prob"],
            "initializer_range": config["initializer_range"],
        }
        return model_config

    def _skip_weight(self, mapping, name):
        mapping.pop(name)
        logging.info("Skip load weight: %s", name)
