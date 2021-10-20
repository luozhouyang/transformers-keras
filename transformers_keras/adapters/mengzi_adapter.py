import logging
import os

from .abstract_adapter import AbstractBertAdapter


class BertAdapterForLangboatMengzi(AbstractBertAdapter):
    """Bert adapter for https://github.com/Langboat/Mengzi"""

    def __init__(
        self,
        pt_bert_prefix="bert",
        pt_mlm_prefix="cls.predictions",
        pt_nsp_prefix="cls.seq_relationship",
        pt_sop_prefix="cls.seq_relationship",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pt_bert_prefix = pt_bert_prefix
        self.pt_mlm_prefix = pt_mlm_prefix
        self.pt_nsp_prefix = pt_nsp_prefix
        self.pt_sop_prefix = pt_sop_prefix

    def _parse_files(self, model_path, **kwargs):
        if not os.path.exists(model_path):
            raise ValueError("model_path: {} does not exist.".format(model_path))
        model_path = os.path.abspath(model_path)
        files = {
            "config_file": os.path.join(model_path, "config.json"),
            "model_bin": os.path.join(model_path, "pytorch_model.bin"),
            "vocab_file": os.path.join(model_path, "vocab.txt"),
        }
        return files

    def _read_pretrained_weights(self, model_path, **kwargs):
        try:
            import torch
        except Exception as e:
            raise ValueError("You must install torch when loading pretrained weights from Langboat/Mengzi.")
        if self.model_files is None:
            self.model_files = self._parse_files(model_path, **kwargs)
        model_bin = self.model_files["model_bin"]
        model = torch.load(model_bin, map_location=kwargs.get("map_location", torch.device("cpu")))
        torch_weights_map = {}
        for k in sorted(model.keys()):
            torch_weights_map[k] = model[k].numpy()
        return torch_weights_map

    def _adapte_backbone_weights(self, model, model_config, **kwargs):
        mapping = {}
        model_prefix = model.bert_model.name if self.use_functional_api else model.name + "/" + model.bert_model.name
        torch_prefix = self.pt_bert_prefix
        logging.info("Adapting bert backbone weights, using model bert prefix: %s", model_prefix)
        logging.info("Adapting bert backbone weights, using torch bert prefix: %s", torch_prefix)

        def _add_weight(name):
            k = model_prefix + "/" + name
            v = str(name.rstrip(":0")).replace("/", ".")
            if "_embeddings" in v:
                v += ".weight"
            v = v.replace("layer_", "layer.")
            v = v.replace("kernel", "weight")
            v = v.replace("bias", "bias")
            v = v.replace("gamma", "weight")
            v = v.replace("beta", "bias")
            mapping[k] = torch_prefix + "." + v

        _add_weight("embeddings/word_embeddings:0")
        _add_weight("embeddings/position_embeddings:0")
        _add_weight("embeddings/token_type_embeddings:0")
        _add_weight("embeddings/LayerNorm/gamma:0")
        _add_weight("embeddings/LayerNorm/beta:0")

        for i in range(model_config["num_layers"]):
            # self attention
            for x in ["query", "key", "value"]:
                _add_weight("encoder/layer_{}/attention/self/{}/kernel:0".format(i, x))
                _add_weight("encoder/layer_{}/attention/self/{}/bias:0".format(i, x))
            _add_weight("encoder/layer_{}/attention/output/dense/kernel:0".format(i))
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
        model_prefix = "cls/predictions" if self.use_functional_api else model.name + "/cls/predictions"
        torch_prefix = self.pt_mlm_prefix
        logging.info("Adapting mlm weights, using model mlm prefix: %s", model_prefix)
        logging.info("Adapting mlm weights, using torch mlm prefix: %s", torch_prefix)
        mapping = {
            "transform/dense/kernel:0": "transform.dense.weight",
            "transform/dense/bias:0": "transform.dense.bias",
            "transform/LayerNorm/gamma:0": "transform.LayerNorm.weight",
            "transform/LayerNorm/beta:0": "transform.LayerNorm.bias",
            "output_bias:0": "bias",
        }

        # add name prefix
        for k in list(mapping.keys()):
            kk = model_prefix + "/" + k
            vv = torch_prefix + "." + mapping.pop(k)
            mapping[kk] = vv

        return mapping

    def _adapte_nsp_weights(self, model, model_config, **kwargs):
        if not self.with_nsp:
            return {}
        # TODO: impl nsp weights loadding
        logging.info("Load nsp weights is not supported yet. You can subclass this adapter to implement it.")
        return {}

    def _adapte_sop_weights(self, model, model_config, **kwargs):
        if not self.with_sop:
            return {}
        # TODO: impl sop weights loadding
        logging.info("Load sop weights is not supported yet. You can subclass this adapter to implement it.")
        return {}

    def _zip_weights(self, model, model_config, weights_mapping, **kwargs):
        zipping_weights, zipping_values = [], []
        transpose_keys= [
            "output.dense.weight", 
            "intermediate.dense.weight", 
            "self.key.weight", 
            "self.query.weight", 
            "self.value.weight",
            "pooler.dense.weight",
        ]
        for m in model.trainable_weights:
            name = m.name
            if name in self.weights_to_skip:
                continue
            if name not in weights_mapping:
                logging.warning("Model weight not in weights mapping: %s", name)
                continue
            zipping_weights.append(m)
            torch_key = weights_mapping[name]
            torch_value = self._pretrained_weights_map[torch_key]
            if any(str(torch_key).endswith(x) for x in transpose_keys):
                torch_value = torch_value.T
                self._pretrained_weights_map[torch_key] = torch_value
            zipping_values.append(torch_value)
        return zipping_weights, zipping_values
