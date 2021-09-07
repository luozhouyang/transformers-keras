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
from transformers_keras.question_answering.callback import (
    BaseMetricForQuestionAnswering,
    ExactMatchForQuestionAnswering,
    F1ForQuestionAnswering,
)
from transformers_keras.question_answering.dataset import (
    QuestionAnsweringDataset,
    QuestionAnsweringExample,
    QuestionAnsweringXDataset,
    QuestionAnsweringXExample,
)
from transformers_keras.question_answering.models import (
    AlbertForQuestionAnswering,
    AlbertForQuestionAnsweringX,
    BertForQuestionAnswering,
    BertForQuestionAnsweringX,
)
from transformers_keras.sentence_embedding.bert_embedding import BertForSentenceEmbedding
from transformers_keras.sentence_embedding.callback import SpearmanForSentenceEmbedding
from transformers_keras.sentence_embedding.simcse_dataset import (
    HardNegativeSimCSEDataset,
    HardNegativeSimCSEExample,
    SupervisedSimCSEDataset,
    SupervisedSimCSEExample,
    UnsupervisedSimCSEDataset,
    UnsupervisedSimCSEExample,
)
from transformers_keras.sentence_embedding.simcse_models import HardNegativeSimCSE, SupervisedSimCSE, UnsupervisedSimCSE
from transformers_keras.sentiment_analysis.ate import BertForAspectTermExtraction
from transformers_keras.sentiment_analysis.dataset import (
    AspectTermExtractionDataset,
    AspectTermExtractionExample,
    OpinionTermExtractionAndClassificationDataset,
)
from transformers_keras.sentiment_analysis.otec import BertForOpinionTermExtractionAndClassification
from transformers_keras.sequence_classification.dataset import SequenceClassificationDataset, SequenceClassificationExample
from transformers_keras.sequence_classification.models import AlbertForSequenceClassification, BertForSequenceClassification
from transformers_keras.token_classification.callback import SeqEvalForTokenClassification
from transformers_keras.token_classification.crf_models import (
    AlertCRFForTokenClassification,
    BertCRFForTokenClassification,
    CRFModel,
)
from transformers_keras.token_classification.dataset import TokenClassificationDataset, TokenClassificationExample
from transformers_keras.token_classification.models import AlbertForTokenClassification, BertForTokenClassification

__name__ = "transformers_keras"
__version__ = "0.4.5"

logging.basicConfig(format="%(asctime)s %(levelname)7s %(filename)20s %(lineno)4d] %(message)s", level=logging.INFO)
