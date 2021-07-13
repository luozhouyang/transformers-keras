import logging

import tensorflow as tf

from .modeling_albert import (Albert, AlbertEmbedding, AlbertEncoder,
                              AlbertEncoderGroup, AlbertEncoderLayer)
from .modeling_bert import (Bert, BertEmbedding, BertEncoder, BertEncoderLayer,
                            BertIntermediate, BertPretrainedModel)
from .modeling_utils import complete_inputs

__name__ = 'transformers_keras'
__version__ = '0.3.1'

logging.basicConfig(format="%(asctime)s %(levelname)7s %(filename)15s %(lineno)4d] %(message)s", level=logging.INFO)
