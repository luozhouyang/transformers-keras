import logging

import tensorflow as tf

from transformers_keras.datasets.abstract_dataset_builder import AbstractDatasetBuilder
from transformers_keras.datasets.bert_dataset_builder import BertTFRecordDatasetBuilder
from transformers_keras.datasets.transformer_dataset_builder import (
    TransformerTextFileDatasetBuilder,
    TransformerTFRecordDatasetBuilder,
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
__version__ = '0.1.4'

logging.basicConfig(format="%(asctime)s %(levelname)s %(filename)15s %(lineno)4d] %(message)s", level=logging.INFO)
