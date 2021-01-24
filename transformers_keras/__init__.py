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

__name__ = 'transformers_keras'
__version__ = '0.2.2'

logging.basicConfig(format="%(asctime)s %(levelname)s %(filename)15s %(lineno)4d] %(message)s", level=logging.INFO)
