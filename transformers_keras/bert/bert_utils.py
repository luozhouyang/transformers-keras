import tensorflow as tf

from transformers_keras.bert.bert_models import Bert4PreTrainingModel


def build_bert_for_pretraining_model(config, training=True, **kwargs):
    input_ids = tf.keras.layers.Input(shape=(config.max_sequence_length,), dtype=tf.int32, name='input_ids')
    input_mask = tf.keras.layers.Input(shape=(config.max_sequence_length,), dtype=tf.int32, name='input_mask')
    position_ids = tf.keras.layers.Input(shape=(config.max_sequence_length,), dtype=tf.int32, name='position_ids')
    token_type_ids = tf.keras.layers.Input(shape=(config.max_sequence_length,), dtype=tf.int32, name='token_type_ids')

    outputs = Bert4PreTrainingModel(config)(
        inputs=[input_ids, position_ids, token_type_ids, input_mask],
        training=training)
    prediction_scores, relation_scores, pooled_output, all_hidden_states, all_attention_scores = outputs
    predictions = tf.keras.layers.Lambda(lambda x: x, name='predictions')(prediction_scores)
    relations = tf.keras.layers.Lambda(lambda x: x, name='relations')(relation_scores)

    model = tf.keras.Model(inputs={
        'input_ids': input_ids,
        'input_mask': input_mask,
        'position_ids': position_ids,
        'token_type_ids': token_type_ids,
    }, outputs={
        'predictions': predictions,
        'relations': relations,
        'pooled_output': pooled_output,
        'all_hidden_states': all_hidden_states,
        'all_attention_scores': all_attention_scores
    })

    model.compile(
        optimizer='adam',
        loss={
            'predictions': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            'relations': tf.keras.losses.BinaryCrossentropy(from_logits=True)
        },
        metrics={
            'predictions': ['acc'],
            'relations': ['acc']
        })
    return model
