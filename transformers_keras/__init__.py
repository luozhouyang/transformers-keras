import logging

import tensorflow as tf

from .callbacks import SavedModelExporter, TransformerLearningRate
from .layers import (DecoderLayer, EncoderLayer, MultiHeadAttention,
                     PointWiseFeedForwardNetwork, ScaledDotProductAttention)
from .losses import MaskedSparseCategoricalCrossentropy
from .metrics import MaskedSparseCategoricalAccuracy
from .modeling_albert import (Albert, AlbertEmbedding, AlbertEncoder,
                              AlbertEncoderGroup, AlbertEncoderLayer,
                              AlbertMLMHead, AlbertSOPHead)
from .modeling_bert import (Bert, BertEmbedding, BertEncoder, BertEncoderLayer,
                            BertIntermediate, BertMLMHead, BertNSPHead)
from .modeling_utils import complete_inputs
from .tokenizers.bert_tokenizer import BertTokenizer
from .tokenizers.tokenizer import BasicTokenizer, WordpieceTokenizer

__name__ = 'transformers_keras'
__version__ = '0.2.4'

logging.basicConfig(format="%(asctime)s %(levelname)7s %(filename)15s %(lineno)4d] %(message)s", level=logging.INFO)
