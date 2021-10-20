from .abstract_adapter import AbstractAdapter
from .albert_adapter import AlbertAdapter
from .bert_adapter import BertAdapter
from .mengzi_adapter import BertAdapterForLangboatMengzi

LANGBOAT_MENGZI_MODELS = set(["langboat/mengzi"])
DEFAULT_BERT_MODELS = {"google/bert", "ymcui/roberta", "ymcui/bert"}


class BertAdapterFactory:
    """Bert adapter factory"""

    @classmethod
    def get(cls, name_or_adapter, **kwargs):
        if isinstance(name_or_adapter, AbstractAdapter):
            return name_or_adapter
        assert isinstance(name_or_adapter, str), "Invalid `name_or_adapter`"
        name = str(name_or_adapter).lower()
        if name in LANGBOAT_MENGZI_MODELS:
            return BertAdapterForLangboatMengzi(
                use_functional_api=kwargs.pop("use_functional_api", True),
                with_mlm=kwargs.pop("with_mlm", False),
                with_nsp=kwargs.pop("with_nsp", False),
                with_sop=kwargs.pop("with_sop", False),
                pt_bert_prefix=kwargs.pop("pt_bert_prefix", "bert"),
                pt_mlm_prefix=kwargs.pop("pt_mlm_prefix", "cls.predictions"),
                pt_nsp_prefix=kwargs.pop("pt_nsp_prefix", "cls.seq_relations"),
                pt_sop_predix=kwargs.pop("pt_sop_prefix", "cls.seq_relations"),
                skip_token_embedding=kwargs.pop("skip_token_embedding", False),
                skip_position_embedding=kwargs.pop("skip_position_embedding", False),
                skip_segment_embedding=kwargs.pop("skip_segment_embedding", False),
                skip_embedding_layernorm=kwargs.pop("skip_embedding_layernorm", False),
                skip_pooler=kwargs.pop("skip_pooler", False),
                check_weights=kwargs.pop("check_weights", True),
                verbose=kwargs.pop("verbose", True),
                **kwargs
            )

        # default bert adapter
        return BertAdapter(
            use_functional_api=kwargs.pop("use_functional_api", True),
            with_mlm=kwargs.pop("with_mlm", False),
            with_nsp=kwargs.pop("with_nsp", False),
            with_sop=kwargs.pop("with_sop", False),
            tf_bert_prefix=kwargs.pop("tf_bert_prefix", "bert"),
            tf_mlm_prefix=kwargs.pop("tf_mlm_prefix", "cls/predictions"),
            tf_nsp_prefix=kwargs.pop("tf_nsp_prefix", "cls/seq_relations"),
            tf_sop_predix=kwargs.pop("tf_sop_prefix", "cls/seq_relations"),
            skip_token_embedding=kwargs.pop("skip_token_embedding", False),
            skip_position_embedding=kwargs.pop("skip_position_embedding", False),
            skip_segment_embedding=kwargs.pop("skip_segment_embedding", False),
            skip_embedding_layernorm=kwargs.pop("skip_embedding_layernorm", False),
            skip_pooler=kwargs.pop("skip_pooler", False),
            check_weights=kwargs.pop("check_weights", True),
            verbose=kwargs.pop("verbose", True),
            **kwargs
        )


class AlbertAdapterFactory:
    """Albert adapter factory"""

    @classmethod
    def get(cls, name_or_adapter, **kwargs):
        if isinstance(name_or_adapter, AbstractAdapter):
            return name_or_adapter
        assert isinstance(name_or_adapter, str), "Invalid `name_or_adapter`!"
        # use AlbertAdapter in default
        return AlbertAdapter(
            use_functional_api=kwargs.pop("use_functional_api", True),
            with_mlm=kwargs.pop("with_mlm", False),
            with_nsp=kwargs.pop("with_nsp", False),
            with_sop=kwargs.pop("with_sop", False),
            tf_bert_prefix=kwargs.pop("tf_bert_prefix", "bert"),
            tf_mlm_prefix=kwargs.pop("tf_mlm_prefix", "cls/predictions"),
            tf_nsp_prefix=kwargs.pop("tf_nsp_prefix", "cls/seq_relations"),
            tf_sop_predix=kwargs.pop("tf_sop_prefix", "cls/seq_relations"),
            skip_embedding_mapping_in=kwargs.pop("skip_embedding_mapping_in", False),
            skip_token_embedding=kwargs.pop("skip_token_embedding", False),
            skip_position_embedding=kwargs.pop("skip_position_embedding", False),
            skip_segment_embedding=kwargs.pop("skip_segment_embedding", False),
            skip_embedding_layernorm=kwargs.pop("skip_embedding_layernorm", False),
            skip_pooler=kwargs.pop("skip_pooler", False),
            skip_mlm=kwargs.pop("skip_mlm", False),
            check_weights=kwargs.pop("check_weights", True),
            verbose=kwargs.pop("verbose", True),
            **kwargs
        )
