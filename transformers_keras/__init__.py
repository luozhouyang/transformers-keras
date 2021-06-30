import logging

import tensorflow as tf

from .callbacks import SavedModelExporter, TransformerLearningRate
from .layers import (DecoderLayer, EncoderLayer, MultiHeadAttention,
                     PointWiseFeedForwardNetwork, ScaledDotProductAttention)
from .losses import MaskedSparseCategoricalCrossentropy
from .metrics import MaskedSparseCategoricalAccuracy
from .modeling_albert import (Albert, AlbertEmbedding, AlbertEncoder,
                              AlbertEncoderGroup, AlbertEncoderLayer)
from .modeling_bert import (Bert, BertEmbedding, BertEncoder, BertEncoderLayer,
                            BertIntermediate)
from .modeling_utils import complete_inputs

__name__ = 'transformers_keras'
__version__ = '0.3.0'

logging.basicConfig(format="%(asctime)s %(levelname)7s %(filename)15s %(lineno)4d] %(message)s", level=logging.INFO)
