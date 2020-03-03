import tensorflow as tf

from transformers_keras.bert.bert_models import Bert4PreTrainingModel
from transformers_keras.losses import MaskedSparseCategoricalCrossentropy
from transformers_keras.metrics import MaskedSparseCategoricalAccuracy


def build_bert_for_pretraining_model(config, training=True, **kwargs):
    input_ids = tf.keras.layers.Input(shape=(config.max_sequence_length,), dtype=tf.int32, name='input_ids')
    input_mask = tf.keras.layers.Input(shape=(config.max_sequence_length,), dtype=tf.int32, name='input_mask')
    token_type_ids = tf.keras.layers.Input(shape=(config.max_sequence_length,), dtype=tf.int32, name='token_type_ids')

    outputs = Bert4PreTrainingModel(config)(
        inputs=[input_ids, token_type_ids, input_mask],
        training=training)
    prediction_scores, relation_scores, pooled_output, all_hidden_states, all_attention_scores = outputs
    predictions = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x), name='predictions')(prediction_scores)
    relations = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x), name='relations')(relation_scores)
    pooled_output = tf.keras.layers.Lambda(lambda x: x, name='pooled_output')(pooled_output)
    all_hidden_states = tf.keras.layers.Lambda(lambda x: x, name='all_hidden_states')(all_hidden_states)
    all_attention_scores = tf.keras.layers.Lambda(lambda x: x, name='all_attention_scores')(all_attention_scores)

    model = tf.keras.Model(
        inputs=[input_ids, input_mask, token_type_ids],
        outputs=[predictions, relations, pooled_output, all_hidden_states, all_attention_scores])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-8, clipnorm=1.0),
        loss={
            # 'predictions': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # TODO masking
            'predictions': MaskedSparseCategoricalCrossentropy(from_logits=False),
            'relations': tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        },
        metrics={
            'predictions': [
                # tf.keras.metrics.SparseCategoricalAccuracy(),
                MaskedSparseCategoricalAccuracy(name='masked_lm_accuracy')
            ],
            'relations': [
                tf.keras.metrics.CategoricalAccuracy(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ]
        })
    return model
