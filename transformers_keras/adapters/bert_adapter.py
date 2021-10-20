import logging

import tensorflow as tf

from .abstract_adapter import AbstractBertAdapter


class BertAdapterForTensorFlow(AbstractBertAdapter):
    """Bert adapter for tensorflow checkpoints"""

    def __init__(
        self,
        tf_bert_prefix="bert",
        tf_mlm_prefix="cls/predictions",
        tf_nsp_prefix="cls/seq_relationship",
        tf_sop_prefix="cls/seq_relatoinship",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tf_bert_prefix = tf_bert_prefix
        self.tf_mlm_prefix = tf_mlm_prefix
        self.tf_nsp_prefix = tf_nsp_prefix
        self.tf_sop_pefix = tf_sop_prefix

    def _adapte_backbone_weights(self, model, model_config, **kwargs):
        self_prefix = model.bert_model.name if self.use_functional_api else model.name + "/" + model.bert_model.name
        ckpt_prefix = self.tf_bert_prefix
        logging.info("Adapting bert backbone weights, using model prefix: %s", self_prefix)
        logging.info("Adapting bert backbone weights, using  ckpt prefix: %s", ckpt_prefix)
        assert self_prefix, "self_prefix must not be empty or null!"
        assert ckpt_prefix, "ckpt_prefix must not be empty or null!"

        mapping = {}

        def _add_weight(name):
            k = self_prefix + "/" + name
            v = ckpt_prefix + "/" + name.rstrip(":0")
            mapping[k] = v

        # embeddings
        _add_weight("embeddings/word_embeddings:0")
        _add_weight("embeddings/position_embeddings:0")
        _add_weight("embeddings/token_type_embeddings:0")
        _add_weight("embeddings/LayerNorm/gamma:0")
        _add_weight("embeddings/LayerNorm/beta:0")
        # encoder layers
        for i in range(model_config["num_layers"]):
            # self attention
            for x in ["query", "key", "value"]:
                _add_weight("encoder/layer_{}/attention/self/{}/kernel:0".format(i, x))
                _add_weight("encoder/layer_{}/attention/self/{}/bias:0".format(i, x))
            _add_weight("encoder/layer_{}/attention/output/dense/kernel:0".format(i, x))
            _add_weight("encoder/layer_{}/attention/output/dense/bias:0".format(i))
            _add_weight("encoder/layer_{}/attention/output/LayerNorm/gamma:0".format(i))
            _add_weight("encoder/layer_{}/attention/output/LayerNorm/beta:0".format(i))

            # intermediate
            _add_weight("encoder/layer_{}/intermediate/dense/kernel:0".format(i))
            _add_weight("encoder/layer_{}/intermediate/dense/bias:0".format(i))

            # output
            _add_weight("encoder/layer_{}/output/dense/kernel:0".format(i))
            _add_weight("encoder/layer_{}/output/dense/bias:0".format(i))
            _add_weight("encoder/layer_{}/output/LayerNorm/gamma:0".format(i))
            _add_weight("encoder/layer_{}/output/LayerNorm/beta:0".format(i))

        # pooler
        _add_weight("pooler/dense/kernel:0")
        _add_weight("pooler/dense/bias:0")

        return mapping

    def _adapte_mlm_weights(self, model, model_config, **kwargs):
        if not self.with_mlm:
            return {}
        self_mlm_prefix = "cls/predictions" if self.use_functional_api else model.name + "/cls/predictions"
        ckpt_mlm_prefix = self.tf_mlm_prefix
        logging.info("Adapting bert mlm weights, using model mlm prefix: %s", self_mlm_prefix)
        logging.info("Adapting bert mlm weights, using  ckpt mlm prefix: %s", ckpt_mlm_prefix)
        assert self_mlm_prefix, "self_mlm_prefix must not be empty or null!"
        assert ckpt_mlm_prefix, "ckpt_mlm_prefix must not be empty or null!"

        mapping = {}

        def _add_weight(name):
            k = self_mlm_prefix + "/" + name
            v = ckpt_mlm_prefix + "/" + name.rstrip(":0")
            mapping[k] = v

        _add_weight("transform/dense/kernel:0")
        _add_weight("transform/dense/bias:0")
        _add_weight("transform/LayerNorm/gamma:0")
        _add_weight("transform/LayerNorm/beta:0")
        _add_weight("output_bias:0")

        return mapping

    def _adapte_nsp_weights(self, model, model_config, **kwargs):
        if not self.with_nsp:
            return {}
        self_nsp_prefix = "cls/seq_relationship" if self.use_functional_api else model.name + "/cls/seq_relationship"
        ckpt_nsp_prefix = self.tf_nsp_prefix
        logging.info("Adapting bert nsp weights, using model nsp prefix: %s", self_nsp_prefix)
        logging.info("Adapting bert nsp weights, using  ckpt nsp prefix: %s", ckpt_nsp_prefix)
        assert self_nsp_prefix, "self_nsp_prefix must not be empty or null!"
        assert ckpt_nsp_prefix, "ckpt_nsp_prefix must not be empty or null!"

        mappings = {}

        def _add_weight(name):
            k = self_nsp_prefix + "/" + name + ":0"
            v = ckpt_nsp_prefix + "/" + name
            mappings[k] = v

        weights = [
            w for (w, _) in tf.train.list_variables(self.model_files["ckpt"]) if str(w).startswith(ckpt_nsp_prefix)
        ]
        for w in weights:
            _add_weight(w[len(ckpt_nsp_prefix) :])

        return mappings

    def _adapte_sop_weights(self, model, model_config, **kwargs):
        if not self.with_sop:
            return {}
        self_sop_prefix = "cls/seq_relationship" if self.use_functional_api else model.name + "/cls/seq_relationship"
        ckpt_sop_prefix = self.tf_sop_prefix
        logging.info("Adapting bert sop weights, using model sop prefix: %s", self_sop_prefix)
        logging.info("Adapting bert sop weights, using  ckpt sop prefix: %s", ckpt_sop_prefix)
        assert self_sop_prefix, "self_sop_prefix must not be empty or null!"
        assert ckpt_sop_prefix, "ckpt_sop_prefix must not be empty or null!"

        mappings = {}

        def _add_weight(name):
            k = self_sop_prefix + "/" + name + ":0"
            v = ckpt_sop_prefix + "/" + name
            mappings[k] = v

        weights = [
            w for (w, _) in tf.train.list_variables(self.model_files["ckpt"]) if str(w).startswith(ckpt_sop_prefix)
        ]
        for w in weights:
            _add_weight(w[len(ckpt_sop_prefix) :])

        return mappings

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


class BertAdapter(BertAdapterForTensorFlow):
    """Default bert adapter"""

    pass
