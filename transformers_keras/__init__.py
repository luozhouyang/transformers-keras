import logging

from smile_datasets import (
    DatapipeForHardNegativeSimCSE,
    DatapipeForMaksedLanguageModel,
    DatapipeForQuestionAnswering,
    DatapipeForSequenceClassifiation,
    DatapipeForSupervisedSimCSE,
    DatapipeForTokenClassification,
    DatapipeForUnsupervisedSimCSE,
    DatasetForHardNegativeSimCSE,
    DatasetForMaskedLanguageModel,
    DatasetForQuestionAnswering,
    DatasetForSequenceClassification,
    DatasetForSupervisedSimCSE,
    DatasetForTokenClassification,
    DatasetForUnsupervisedSimCSE,
    ExampleForHardNegativeSimCSE,
    ExampleForMaskedLanguageModel,
    ExampleForQuestionAnswering,
    ExampleForSequenceClassification,
    ExampleForSupervisedSimCSE,
    ExampleForTokenClassification,
    ExampleForUnsupervisedSimCSE,
    ParserForHardNegSimCSE,
    ParserForMaskedLanguageModel,
    ParserForQuestionAnswering,
    ParserForSequenceClassification,
    ParserForSupervisedSimCSE,
    ParserForTokenClassification,
    ParserForUnsupervisedSimCSE,
)

from transformers_keras.adapters.abstract_adapter import (
    AbstractAdapter,
    AbstractAlbertAdapter,
    AbstractBertAdapter,
    BaseAdapter,
)
from transformers_keras.adapters.adapter_factory import AlbertAdapterFactory, BertAdapterFactory
from transformers_keras.adapters.albert_adapter import AlbertAdapter, AlbertAdapterForTensorFlow
from transformers_keras.adapters.bert_adapter import BertAdapter, BertAdapterForTensorFlow
from transformers_keras.adapters.mengzi_adapter import BertAdapterForLangboatMengzi
from transformers_keras.common.metrics import ExactMatch, F1ForSequence
from transformers_keras.distiller import Distiller
from transformers_keras.lm.mlm import BertForMaskedLanguageModel
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
    EMForQuestionAnswering,
    F1ForQuestionAnswering,
)
from transformers_keras.question_answering.models import (
    AlbertForQuestionAnswering,
    AlbertForQuestionAnsweringX,
    BertForQuestionAnswering,
    BertForQuestionAnsweringX,
)
from transformers_keras.sentence_embedding.bert_embedding import BertForSentenceEmbedding
from transformers_keras.sentence_embedding.callback import SpearmanForSentenceEmbedding
from transformers_keras.sentence_embedding.simcse_models import HardNegativeSimCSE, SupervisedSimCSE, UnsupervisedSimCSE
from transformers_keras.sentiment_analysis.ate import BertForAspectTermExtraction
from transformers_keras.sentiment_analysis.otec import BertForOpinionTermExtractionAndClassification
from transformers_keras.sequence_classification.models import (
    AlbertForSequenceClassification,
    BertForSequenceClassification,
)
from transformers_keras.token_classification.callback import SeqEvalForTokenClassification
from transformers_keras.token_classification.crf_models import (
    AlertCRFForTokenClassification,
    BertCRFForTokenClassification,
    CRFModel,
)
from transformers_keras.token_classification.models import AlbertForTokenClassification, BertForTokenClassification

__name__ = "transformers_keras"
__version__ = "0.5.1"

logging.basicConfig(format="%(asctime)s %(levelname)7s %(filename)20s %(lineno)4d] %(message)s", level=logging.INFO)
