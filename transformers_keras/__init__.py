import logging

import tensorflow as tf

from .modeling_albert import (Albert, AlbertEmbedding, AlbertEncoder,
                              AlbertEncoderGroup, AlbertEncoderLayer,
                              AlbertModel, AlbertPretrainedModel)
from .modeling_bert import (Bert, BertEmbedding, BertEncoder, BertEncoderLayer,
                            BertIntermediate, BertModel, BertPretrainedModel)
from .modeling_utils import complete_inputs
from .question_answering.models import (AlbertForQuestionAnswering,
                                        BertForQuestionAnswering)
from .sequence_classify.models import (AlbertForSequenceClassification,
                                       BertForSequenceClassification)

__name__ = 'transformers_keras'
__version__ = '0.4.0'

logging.basicConfig(format="%(asctime)s %(levelname)7s %(filename)20s %(lineno)4d] %(message)s", level=logging.INFO)
