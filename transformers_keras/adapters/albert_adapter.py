import logging

import tensorflow as tf

from .abstract_adapter import AbstractAlbertAdapter


class AlbertAdapterForTensorFlow(AbstractAlbertAdapter):
    """Albert adapter for tensorflow"""

    def __init__(
        self,
        tf_albert_prefix="bert",
        tf_mlm_prefix="cls/predictions",
        tf_nsp_prefix="cls/seq_relationship",
        tf_sop_prefix="cls/seq_relationship",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tf_albert_prefix = tf_albert_prefix
        self.tf_mlm_prefix = tf_mlm_prefix
        self.tf_sop_preifx = tf_sop_prefix
        self.tf_nsp_prefix = tf_nsp_prefix

    def _adapte_backbone_weights(self, model, model_config, **kwargs):
        # TODO: impl this more efficient
        return self._adapte_albert_weights_legacy(
            model, self.model_files["ckpt"], model_config, use_functional_api=self.use_functional_api, **kwargs
        )

    def _adapte_mlm_weights(self, model, model_config, **kwargs):
        # TODO: impl this more efficient
        return self._adapte_mlm_weights_legacy(
            model, self.model_files["ckpt"], use_functional_api=self.use_functional_api, **kwargs
        )

    def _adapte_nsp_weights(self, model, model_config, **kwargs):
        logging.info("Adapteing nsp weights is not supported yet. You can subclass this adapter to implement it!")
        return {}

    def _adapte_sop_weights(self, model, model_config, **kwargs):
        # TODO: impl this more efficient
        return self._adapte_sop_weights_legacy(
            model, self.model_files["ckpt"], use_functional_api=self.use_functional_api, **kwargs
        )

    def _zip_weights(self, model, model_config, weights_mapping, **kwargs):
        zipping_weights, zipping_values = [], []
        for m in model.trainable_weights:
            name = m.name
            if name in self.weights_to_skip:
                continue
            if name not in weights_mapping:
                logging.warning("Model weight not in weights mapping: %s", name)
                continue
            zipping_weights.append(m)
            zipping_values.append(self._pretrained_weights_map[weights_mapping[name]])
        return zipping_weights, zipping_values

    def _adapte_mlm_weights_legacy(self, model, ckpt, use_functional_api=True, **kwargs):
        if not self.with_mlm:
            return {}
        self_mlm_prefix = "cls/predictions" if use_functional_api else model.name + "/cls/predictions"
        ckpt_mlm_prefix = self.ckpe_mlm_prefix
        logging.info("Adapting mlm weights, using model mlm prefix: %s", self_mlm_prefix)
        logging.info("Adapting mlm weights, using  ckpt mlm prefix: %s", ckpt_mlm_prefix)
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
        return mapping

    def _adapte_sop_weights_legacy(self, model, ckpt, use_functional_api=True, **kwargs):
        if not self.with_sop:
            return {}
        self_sop_prefix = "cls/seq_relationship" if use_functional_api else model.name + "/cls/seq_relationship"
        ckpt_sop_prefix = self.tf_sop_preifx
        logging.info("Adapting sop weights, using model sop prefix: %s", self_sop_prefix)
        logging.info("Adapting sop weights, using  ckpt sop prefix: %s", ckpt_sop_prefix)
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
        return mapping

    def _adapte_albert_weights_legacy(self, model, ckpt, model_config, use_functional_api=True, **kwargs):
        mapping = {}
        self_albert_prefix = (
            model.albert_model.name if use_functional_api else model.name + "/" + model.albert_model.name
        )
        ckpt_albert_prefix = self.tf_albert_prefix
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


class AlbertAdapter(AlbertAdapterForTensorFlow):
    """Default adapter for albert"""

    pass
