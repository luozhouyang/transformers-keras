import json
import logging

import tensorflow as tf

from .abstract_adapter import AbstractAdapter, zip_weights


class AlbertAdapter(AbstractAdapter):
    def __init__(self, skip_embedding_mapping_in=False, **kwargs):
        super().__init__(**kwargs)
        self.skip_embedding_mapping_in = skip_embedding_mapping_in

    def adapte_config(self, config_file, **kwargs):
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

    def adapte_weights(self, model, ckpt, model_config, use_functional_api=True, **kwargs):
        ckpt_weight_names = [x[0] for x in tf.train.list_variables(ckpt)]
        self_weight_names = set([x.name for x in model.trainable_weights])
        weight_mapping = {}
        albert_weight_mapping = self._adapte_albert_weights(
            model, ckpt, model_config, use_functional_api=use_functional_api, **kwargs
        )
        weight_mapping.update(albert_weight_mapping)

        mlm_weight_mapping = self._adapte_mlm_weights(model, ckpt, use_functional_api=use_functional_api, **kwargs)
        weight_mapping.update(mlm_weight_mapping)

        sop_weight_mapping = self._adapte_sop_weights(model, ckpt, use_functional_api=use_functional_api, **kwargs)
        weight_mapping.update(sop_weight_mapping)

        if kwargs.get("check_weights", False):
            self._check_weights(weight_mapping, model, ckpt)

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

    def _adapte_sop_weights(self, model, ckpt, use_functional_api=True, **kwargs):
        with_sop = kwargs.get("with_sop", False)
        if not with_sop:
            logging.info("Skipping to adapte weights for SOP due to option `with_sop` set to `False`")
            return {}
        self_sop_prefix = "cls/seq_relationship" if use_functional_api else model.name + "/cls/seq_relationship"
        ckpt_sop_prefix = kwargs.get("ckpt_sop_prefix", "cls/seq_relationship")
        logging.info("Adapting SOP weights, using model weight prefix: %s", self_sop_prefix)
        logging.info("Adapting SOP weights, using  ckpt weight prefix: %s", ckpt_sop_prefix)
        ckpt_weight_names = [x for (x, _) in tf.train.list_variables(ckpt) if str(x).startswith(ckpt_sop_prefix)]
        self_weight_names = set([x.name for x in model.trainable_weights if str(x.name).startswith(self_sop_prefix)])
        mapping = {}
        for w in ckpt_weight_names:
            # TODO skip unused variables
            if str(w).startswith(ckpt_sop_prefix):
                mw = self_sop_prefix + w.lstrip(ckpt_sop_prefix) + ":0"
                if mw not in self_weight_names:
                    logging.warning("weight: %s not in model weights", mw)
                    continue
                mapping[mw] = w
        # zip weight names and values
        zipped_weights = zip_weights(model, ckpt, mapping, self_weight_names, **kwargs)
        # set values to weights
        tf.keras.backend.batch_set_value(zipped_weights)

        return mapping

    def _adapte_albert_weights(self, model, ckpt, model_config, use_functional_api=True, **kwargs):
        mapping = {}
        self_albert_prefix = (
            model.albert_model.name if use_functional_api else model.name + "/" + model.albert_model.name
        )
        ckpt_albert_prefix = kwargs.get("ckpt_albert_prefix", "bert")
        logging.info("Adapting albert weights, using model weight prefix: %s", self_albert_prefix)
        logging.info("Adapting albert weights, using  ckpt weight prefix: %s", ckpt_albert_prefix)
        self_weight_names = set([x.name for x in model.trainable_weights if str(x.name).startswith(self_albert_prefix)])
        ckpt_weight_names = [x for (x, _) in tf.train.list_variables(ckpt) if str(x).startswith(ckpt_albert_prefix)]

        embedding_mapping = self._adapte_embedding_weights(
            model, ckpt, self_weight_names, ckpt_weight_names, use_functional_api=use_functional_api, **kwargs
        )
        mapping.update(embedding_mapping)

        encoder_mapping = self._adapte_encoder_weights(
            model,
            ckpt,
            self_weight_names,
            ckpt_weight_names,
            num_groups=model_config["num_groups"],
            num_layers_each_group=model_config["num_layers_each_group"],
            **kwargs
        )
        mapping.update(encoder_mapping)

        # skip weights
        self_albert_prefix = (
            model.albert_model.name if use_functional_api else model.name + "/" + model.albert_model.name
        )
        self._skip_weights(mapping, self_albert_prefix)

        # zip weight names and values
        zipped_weights = zip_weights(model, ckpt, mapping, self_weight_names, **kwargs)
        # set values to weights
        tf.keras.backend.batch_set_value(zipped_weights)

        # check weights
        if kwargs.get("check_weights", False):
            self._check_weights(mapping, model, ckpt, **kwargs)

        return mapping

    def _adapte_embedding_weights(
        self, model, ckpt, self_weight_names, ckpt_weight_names, use_functional_api=True, **kwargs
    ):
        model_prefix = model.albert_model.name if use_functional_api else model.name + "/" + model.albert_model.name
        ckpt_prefix = kwargs.get("ckpt_albert_prefix", "bert")
        mapping = {}
        for w in ckpt_weight_names:
            # mapping embedding weights
            if any(x in w for x in ["embeddings", "pooler"]):
                mw = model_prefix + w.lstrip(ckpt_prefix) + ":0"
                if mw not in self_weight_names:
                    logging.warning("weight: %s not in model weights", mw)
                    continue
                mapping[mw] = w
            # mapping embedding mapin weights
            if "embedding_hidden_mapping_in/kernel" in w:
                mw = model_prefix + "/encoder/embedding_mapping/kernel:0"
                if mw not in self_weight_names:
                    logging.warning("weight: %s not in model weights", mw)
                    continue
                mapping[mw] = w
            if "embedding_hidden_mapping_in/bias" in w:
                mw = model_prefix + "/encoder/embedding_mapping/bias:0"
                if mw not in self_weight_names:
                    logging.warning("weight: %s not in model weights", mw)
                    continue
                mapping[mw] = w
        return mapping

    def _adapte_encoder_weights(
        self,
        model,
        ckpt,
        self_weight_names,
        ckpt_weight_names,
        num_groups,
        num_layers_each_group,
        use_functional_api=True,
        **kwargs
    ):
        model_prefix = model.albert_model.name if use_functional_api else model.name + "/" + model.albert_model.name
        ckpt_prefix = kwargs.get("ckpt_albert_prefix", "bert")
        mapping = {}
        for group in range(num_groups):
            for layer in range(num_layers_each_group):
                k_prefix = "{}/encoder/group_{}/layer_{}/".format(model_prefix, group, layer)
                v_prefix = "{}/encoder/transformer/group_{}/inner_group_{}/".format(ckpt_prefix, group, layer)
                # attention
                for n in ["query", "key", "value"]:
                    for x in ["kernel", "bias"]:
                        k = k_prefix + "attention/{}/{}:0".format(n, x)
                        v = v_prefix + "attention_1/self/{}/{}".format(n, x)
                        if k not in self_weight_names:
                            logging.warning("weight: %s not in model weights", k)
                            continue
                        mapping[k] = v

                # attention dense
                for n in ["kernel", "bias"]:
                    k = k_prefix + "attention/dense/{}:0".format(n)
                    v = v_prefix + "attention_1/output/dense/{}".format(n)
                    if k not in self_weight_names:
                        logging.warning("weight: %s not in model weights", k)
                        continue
                    mapping[k] = v

                for n in ["gamma", "beta"]:
                    # attention layer norm
                    k = k_prefix + "attention/layer_norm/{}:0".format(n)
                    v = v_prefix + "LayerNorm/{}".format(n)
                    if k not in self_weight_names:
                        logging.warning("weight: %s not in model weights", k)
                        continue
                    mapping[k] = v
                    # albert encoder layer norm
                    k = k_prefix + "layer_norm/{}:0".format(n)
                    v = v_prefix + "LayerNorm_1/{}".format(n)
                    if k not in self_weight_names:
                        logging.warning("weight: %s not in model weights", k)
                        continue
                    mapping[k] = v

                for n in ["kernel", "bias"]:
                    # intermediate
                    k = k_prefix + "ffn/{}:0".format(n)
                    v = v_prefix + "ffn_1/intermediate/dense/{}".format(n)
                    if k not in self_weight_names:
                        logging.warning("weight: %s not in model weights", k)
                        continue
                    mapping[k] = v
                    # dense
                    k = k_prefix + "ffn_output/{}:0".format(n)
                    v = v_prefix + "ffn_1/intermediate/output/dense/{}".format(n)
                    if k not in self_weight_names:
                        logging.warning("weight: %s not in model weights", k)
                        continue
                    mapping[k] = v
        return mapping

    def _skip_weights(self, mapping, model_prefix):
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
