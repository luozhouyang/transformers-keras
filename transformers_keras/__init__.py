import logging

import tensorflow as tf

from transformers_keras.datasets.abstract_dataset_builder import AbstractDatasetBuilder
from transformers_keras.datasets.bert_dataset_builder import BertTFRecordDatasetBuilder
from transformers_keras.datasets.transformer_dataset_builder import (
    TransformerDatasetBuilder,
    TransformerTextFileDatasetBuilder,
    TransformerTFRecordDatasetBuilder,
)
from transformers_keras.tokenizers.bert_tokenizer import BertAbstractTokenizer, BertDefaultTokenizer, BertVocabBasedTokenizer
from transformers_keras.tokenizers.tokenizer import BasicTokenizer, WordpieceTokenizer
from transformers_keras.tokenizers.transformer_tokenizer import (
    TransformerAbstractTokenizer,
    TransformerDefaultTokenizer,
    TransformerJiebaTokenizer,
    TransformerVocabBasedTokenizer,
)

from .callbacks import SavedModelExporter, TransformerLearningRate
from .layers import DecoderLayer, EncoderLayer, MultiHeadAttention, PointWiseFeedForwardNetwork, ScaledDotProductAttention
from .losses import MaskedSparseCategoricalCrossentropy
from .metrics import MaskedSparseCategoricalAccuracy
from .modeling_albert import (
    Albert4PreTraining,
    AlbertEmbedding,
    AlbertEncoder,
    AlbertEncoderGroup,
    AlbertEncoderLayer,
    AlbertMLMHead,
    AlbertModel,
    AlbertSOPHead,
)
from .modeling_bert import Bert, Bert4PreTraining, BertEmbedding, BertMLMHead, BertNSPHead
from .modeling_transformer import PositionalEncoding, Transformer, TransformerDecoder, TransformerEmbedding, TransformerEncoder
from .runners import AlbertRunner, BertRunner, TransformerRunner

__name__ = 'transformers_keras'
r_version__ = '0.1.3'

logging.basicConfig(format="%(asctime)s %(levelname)s %(filename)15s %(lineno)4d] %(message)s", level=logging.INFO)


def build_pretraining_bert_model(model_config):
    max_sequence_length = model_config.get('max_positions', 512)
    input_ids = tf.keras.layers.Input(
        shape=(max_sequence_length,), dtype=tf.int32, name='input_ids')
    input_mask = tf.keras.layers.Input(
        shape=(max_sequence_length,), dtype=tf.int32, name='input_mask')
    segment_ids = tf.keras.layers.Input(
        shape=(max_sequence_length,), dtype=tf.int32, name='segment_ids')

    inputs = (input_ids, segment_ids, input_mask)
    bert = Bert4PreTraining(**model_config)
    outputs = bert(inputs)

    predictions = tf.keras.layers.Lambda(lambda x: x, name='predictions')(outputs[0])
    relations = tf.keras.layers.Lambda(lambda x: x, name='relations')(outputs[1])

    model = tf.keras.Model(inputs=inputs, outputs=[predictions, relations])
    lr = model_config.get('learning_rate', 3e-5)
    epsilon = model_config.get('epsilon', 1e-12)
    clipnorm = model_config.get('clipnorm', 1.0)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=epsilon, clipnorm=clipnorm),
        loss={
            'predictions': MaskedSparseCategoricalCrossentropy(mask_id=0, from_logits=True, name='pred_loss'),
            'relations': tf.keras.losses.CategoricalCrossentropy(from_logits=True, name='rel_loss'),
        },
        metrics={
            'predictions': [
                MaskedSparseCategoricalAccuracy(mask_id=0, from_logits=True, name='pred_acc'),
            ],
            'relations': [
                tf.keras.metrics.CategoricalAccuracy(name='rel_acc'),
            ]
        })
    model.summary()
    return model


def build_pretraining_albert_model(model_config):
    max_sequence_length = model_config.get('max_positions', 512)
    input_ids = tf.keras.layers.Input(
        shape=(max_sequence_length,), dtype=tf.int32, name='input_ids')
    input_mask = tf.keras.layers.Input(
        shape=(max_sequence_length,), dtype=tf.int32, name='input_mask')
    segment_ids = tf.keras.layers.Input(
        shape=(max_sequence_length,), dtype=tf.int32, name='segment_ids')

    inputs = (input_ids, segment_ids, input_mask)
    albert = Albert4PreTraining(**model_config)
    predictions, relations, all_states, all_attn_weights = albert(inputs=inputs)

    predictions = tf.keras.layers.Lambda(lambda x: x, name='predictions')(predictions)
    relations = tf.keras.layers.Lambda(lambda x: x, name='relations')(relations)

    model = tf.keras.Model(
        inputs=[input_ids, segment_ids, input_mask], outputs=[predictions, relations])

    lr = model_config.get('learning_rate', 3e-5)
    epsilon = model_config.get('epsilon', 1e-12)
    clipnorm = model_config.get('clipnorm', 1.0)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=epsilon, clipnorm=clipnorm),
        loss={
            'predictions': MaskedSparseCategoricalCrossentropy(
                mask_id=0, from_logits=True, name='pred_loss'),
            'relations': tf.keras.losses.CategoricalCrossentropy(
                from_logits=True, name='rel_loss'),
        },
        metrics={
            'predictions': [
                MaskedSparseCategoricalAccuracy(
                    mask_id=0, from_logits=False, name='pred_acc'),
            ],
            'relations': [
                tf.keras.metrics.CategoricalAccuracy(name='rel_acc'),
            ]
        })
    model.summary()
    return model
