import json
import logging

import tensorflow as tf

from .abstract_adapter import AbstractAdapter, zip_weights


class BertAdapter(AbstractAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def adapte_config(self, config_file, **kwargs):
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

    def adapte_weights(self, model, ckpt, model_config, use_functional_api=True, **kwargs):
        ckpt_weight_names = [x[0] for x in tf.train.list_variables(ckpt)]
        self_weight_names = set([x.name for x in model.trainable_weights])
        weight_mapping = {}
        bert_weight_mapping = self._adapte_bert_weights(model, ckpt, use_functional_api=use_functional_api, **kwargs)
        weight_mapping.update(bert_weight_mapping)

        mlm_weight_mapping = self._adapte_mlm_weights(model, ckpt, use_functional_api=use_functional_api, **kwargs)
        weight_mapping.update(mlm_weight_mapping)

        nsp_weight_mapping = self._adapte_nsp_weights(model, ckpt, use_functional_api=use_functional_api, **kwargs)
        weight_mapping.update(nsp_weight_mapping)

        check_weights = kwargs.get("check_weights", False)
        if check_weights:
            self._check_weights(weight_mapping, model, ckpt)

    # apdate BERT backbone's weights
    def _adapte_bert_weights(self, model, ckpt, use_functional_api=True, **kwargs):
        model_prefix = model.bert_model.name if use_functional_api else model.name + "/" + model.bert_model.name
        logging.info("Adapting bert weights, using model weight prefix: %s", model_prefix)
        ckpt_prefix = kwargs.get("ckpt_bert_prefix", "bert")
        logging.info("Adapting bert weights, using  ckpt weight prefix: %s", ckpt_prefix)
        ckpt_weight_names = [x for (x, _) in tf.train.list_variables(ckpt) if str(x).startswith(ckpt_prefix)]
        self_weight_names = set([x.name for x in model.trainable_weights if str(x.name).startswith(model_prefix)])
        mapping = {}
        # collect weights mapping
        for w in ckpt_weight_names:
            if any(x in w for x in ["embeddings", "pooler", "encoder"]):
                mw = model_prefix + w.lstrip(ckpt_prefix) + ":0"
                if mw not in self_weight_names:
                    logging.warning("weight: %s not in model weights", mw)
                    continue
                mapping[mw] = w
        # skip weights
        if self.skip_token_embedding:
            self._skip_weight(mapping, model_prefix + "/embeddings/word_embeddings:0")
        if self.skip_position_embedding:
            self._skip_weight(mapping, model_prefix + "/embeddings/position_embeddings:0")
        if self.skip_segment_embedding:
            self._skip_weight(mapping, model_prefix + "/embeddings/token_type_embeddings:0")
        if self.skip_embedding_layernorm:
            self._skip_weight(mapping, model_prefix + "/embeddings/LayerNorm/gamma:0")
            self._skip_weight(mapping, model_prefix + "/embeddings/LayerNorm/beta:0")
        if self.skip_pooler:
            self._skip_weight(mapping, model_prefix + "/pooler/dense/kernel:0")
            self._skip_weight(mapping, model_prefix + "/pooler/dense/bias:0")

        # zip weight names and values
        zipped_weights = zip_weights(model, ckpt, mapping, self_weight_names, **kwargs)
        # set values to weights
        tf.keras.backend.batch_set_value(zipped_weights)

        return mapping

    # adapte MLM's weights
    def _adapte_mlm_weights(self, model, ckpt, use_functional_api=True, **kwargs):
        with_mlm = kwargs.get("with_mlm", False)
        if not with_mlm:
            logging.info("Skipping to adapte weights for MLM due to option `with_mlm` set to `False`")
            return {}
        self_mlm_prefix = "cls/predictions" if use_functional_api else model.name + "/cls/predictions"
        ckpt_mlm_prefix = kwargs.get("ckpt_mlm_prefix", "cls/predictions")
        logging.info("Adapting MLM weights, using model weight prefix: %s", self_mlm_prefix)
        logging.info("Adapting MLM weights, using  ckpt weight prefix: %s", ckpt_mlm_prefix)
        ckpt_weight_names = [x for (x, _) in tf.train.list_variables(ckpt) if str(x).startswith(ckpt_mlm_prefix)]
        self_weight_names = set([x.name for x in model.trainable_weights if str(x.name).startswith(self_mlm_prefix)])
        mapping = {}
        for w in ckpt_weight_names:
            if any(k in w for k in ["key", "value", "query", "prev"]):
                continue
            if str(w).startswith(ckpt_mlm_prefix):
                mw = self_mlm_prefix + w[len(ckpt_mlm_prefix) :] + ":0"
                if mw not in self_weight_names:
                    logging.warning("weight: %s not in model weights", mw)
                    continue
                mapping[mw] = w

        # zip weight names and values
        zipped_weights = zip_weights(model, ckpt, mapping, self_weight_names, **kwargs)
        # set values to weights
        tf.keras.backend.batch_set_value(zipped_weights)

        return mapping

    # adapte NSP's weights
    def _adapte_nsp_weights(self, model, ckpt, use_functional_api=True, **kwargs):
        with_nsp = kwargs.get("with_nsp", False)
        if not with_nsp:
            logging.info("Skipping to adapte weights for NSP due to option `with_nsp` set to `False`")
            return {}
        self_nsp_prefix = "cls/seq_relationship" if use_functional_api else model.name + "/cls/seq_relationship"
        ckpt_nsp_prefix = kwargs.get("ckpt_nsp_prefix", "cls/seq_relationship")
        logging.info("Adapting NSP weights, using model weight prefix: %s", self_nsp_prefix)
        logging.info("Adapting NSP weights, using  ckpt weight prefix: %s", ckpt_nsp_prefix)
        ckpt_weight_names = [x for (x, _) in tf.train.list_variables(ckpt) if str(x).startswith(ckpt_nsp_prefix)]
        self_weight_names = set([x.name for x in model.trainable_weights if str(x.name).startswith(self_nsp_prefix)])
        mapping = {}
        for w in ckpt_weight_names:
            if str(w).startswith(ckpt_nsp_prefix):
                mw = self_nsp_prefix + w.lstrip(ckpt_nsp_prefix) + ":0"
                if mw not in self_weight_names:
                    logging.warning("weight: %s not in model weights", mw)
                    continue
                mapping[mw] = w
        # zip weight names and values
        zipped_weights = zip_weights(model, ckpt, mapping, self_weight_names, **kwargs)
        # set values to weights
        tf.keras.backend.batch_set_value(zipped_weights)

        return mapping
