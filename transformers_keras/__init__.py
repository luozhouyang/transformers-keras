import logging

import tensorflow as tf

from transformers_keras.datasets.abstract_dataset_builder import AbstractDatasetBuilder
from transformers_keras.datasets.bert_dataset_builder import BertTFRecordDatasetBuilder
from transformers_keras.datasets.transformer_dataset_builder import (
    TransformerDatasetBuilder,
    TransformerTextFileDatasetBuilder,
    TransformerTFRecordDatasetBuilder,
)
from transformers_keras.tokenizers.bert_tokenizer import BertAbstractTokenizer, BertDefaultTokenizer, BertVocabBasedTokenizer
from transformers_keras.tokenizers.tokenizer import BasicTokenizer, WordpieceTokenizer
from transformers_keras.tokenizers.transformer_tokenizer import (
    TransformerAbstractTokenizer,
    TransformerDefaultTokenizer,
    TransformerJiebaTokenizer,
    TransformerVocabBasedTokenizer,
)

from .callbacks import SavedModelExporter, TransformerLearningRate
from .layers import DecoderLayer, EncoderLayer, MultiHeadAttention, PointWiseFeedForwardNetwork, ScaledDotProductAttention
from .losses import MaskedSparseCategoricalCrossentropy
from .metrics import MaskedSparseCategoricalAccuracy
from .modeling_albert import (
    Albert,
    AlbertEmbedding,
    AlbertEncoder,
    AlbertEncoderGroup,
    AlbertEncoderLayer,
    AlbertForPretrainingModel,
    AlbertMLMHead,
    AlbertModel,
    AlbertSOPHead,
)
from .modeling_bert import (
    Bert,
    BertEmbedding,
    BertEncoder,
    BertEncoderLayer,
    BertForPretrainingModel,
    BertIntermediate,
    BertMLMHead,
    BertModel,
    BertNSPHead,
)
from .modeling_transformer import PositionalEncoding, Transformer, TransformerDecoder, TransformerEmbedding, TransformerEncoder
from .runners import TransformerRunner

__name__ = 'transformers_keras'
r_version__ = '0.1.3'

logging.basicConfig(format="%(asctime)s %(levelname)s %(filename)15s %(lineno)4d] %(message)s", level=logging.INFO)
