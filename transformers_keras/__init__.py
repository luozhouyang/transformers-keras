import logging

from transformers_keras.datasets.abstract_dataset_builder import AbstractDatasetBuilder
from transformers_keras.datasets.bert_dataset_builder import BertTFRecordDatasetBuilder
from transformers_keras.datasets.transformer_dataset_builder import TransformerDatasetBuilder
from transformers_keras.datasets.transformer_dataset_builder import TransformerTFRecordDatasetBuilder
from transformers_keras.datasets.transformer_dataset_builder import TransformerTextFileDatasetBuilder
from transformers_keras.tokenizers.bert_tokenizer import BertAbstractTokenizer
from transformers_keras.tokenizers.bert_tokenizer import BertDefaultTokenizer
from transformers_keras.tokenizers.bert_tokenizer import BertVocabBasedTokenizer
from transformers_keras.tokenizers.tokenizer import WordpieceTokenizer, BasicTokenizer
from transformers_keras.tokenizers.transformer_tokenizer import TransformerAbstractTokenizer
from transformers_keras.tokenizers.transformer_tokenizer import TransformerDefaultTokenizer
from transformers_keras.tokenizers.transformer_tokenizer import TransformerJiebaTokenizer
from transformers_keras.tokenizers.transformer_tokenizer import TransformerVocabBasedTokenizer
from .callbacks import SavedModelExporter, TransformerLearningRate
from .layers import DecoderLayer
from .layers import EncoderLayer
from .layers import MultiHeadAttention
from .layers import PointWiseFeedForwardNetwork
from .layers import ScaledDotProductAttention
from .losses import MaskedSparseCategoricalCrossentropy
from .metrics import MaskedSparseCategoricalAccuracy
from .modeling_albert import AlbertEmbedding, AlbertEncoder, AlbertEncoderGroup, AlbertMLMHead
from .modeling_albert import AlbertEncoderLayer, Albert4PreTraining, AlbertModel, AlbertSOPHead
from .modeling_bert import BertEmbedding, BertMLMHead, BertNSPHead, Bert4PreTraining, BertModel
from .modeling_transformer import PositionalEncoding, TransformerEmbedding
from .modeling_transformer import TransformerEncoder, TransformerDecoder, Transformer
from .runners import AlbertRunner, BertRunner, TransformerRunner

__name__ = 'transformers_keras'
__version__ = '0.1.0'

logging.basicConfig(format="%(asctime)s %(levelname)s %(filename)15s %(lineno)4d] %(message)s", level=logging.INFO)
