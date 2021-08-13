import logging

import tensorflow as tf

from transformers_keras.modeling_albert import (
    Albert,
    AlbertEmbedding,
    AlbertEncoder,
    AlbertEncoderGroup,
    AlbertEncoderLayer,
    AlbertModel,
    AlbertPretrainedModel,
)
from transformers_keras.modeling_bert import (
    Bert,
    BertEmbedding,
    BertEncoder,
    BertEncoderLayer,
    BertIntermediate,
    BertModel,
    BertPretrainedModel,
)
from transformers_keras.modeling_utils import complete_inputs
from transformers_keras.question_answering.dataset import QuestionAnsweringDataset, QuestionAnsweringExample
from transformers_keras.question_answering.models import AlbertForQuestionAnswering, BertForQuestionAnswering
from transformers_keras.question_answering.parser import (
    AbstractQustionAnsweringExampleParser,
    QuestionAnsweringExampleParserForChinese,
)
from transformers_keras.question_answering.tokenizer import (
    QuestionAnsweringTokenizer,
    QuestionAnsweringTokenizerForChinese,
)
from transformers_keras.sentence_embedding.bert_embedding import BertForSentenceEmbedding
from transformers_keras.sentence_embedding.parser import AbstractSimCSEExampleParser, SimCSEExampleParser
from transformers_keras.sentence_embedding.simcse_dataset import SimCSEDataset, SimCSEExample
from transformers_keras.sentence_embedding.simcse_models import HardNegativeSimCSE, SupervisedSimCSE, UnsupervisedSimCSE
from transformers_keras.sequence_classification.dataset import (
    SequenceClassificationDataset,
    SequenceClassificationExample,
)
from transformers_keras.sequence_classification.models import (
    AlbertForSequenceClassification,
    BertForSequenceClassification,
)
from transformers_keras.sequence_classification.parser import (
    AbstractSequenceClassificationExampleParser,
    SequenceClassificationExampleParser,
)
from transformers_keras.token_classification.crf_models import (
    AlertCRFForTokenClassification,
    BertCRFForTokenClassification,
    CRFModel,
)
from transformers_keras.token_classification.dataset import TokenClassificationDataset, TokenClassificationExample
from transformers_keras.token_classification.models import AlbertForTokenClassification, BertForTokenClassification
from transformers_keras.token_classification.parser import (
    AbstractTokenClassificationExampleParser,
    TokenClassificationExampleParserForChinese,
)
from transformers_keras.token_classification.tokenizer import QuestionAnsweringTokenizerForChinese

__name__ = "transformers_keras"
__version__ = "0.4.2"

logging.basicConfig(format="%(asctime)s %(levelname)7s %(filename)20s %(lineno)4d] %(message)s", level=logging.INFO)
