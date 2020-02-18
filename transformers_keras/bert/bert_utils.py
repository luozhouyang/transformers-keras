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
    prediction_scores, relation_scores, all_hidden_states, all_attention_scores = outputs
    last_hidden_state = tf.keras.layers.Lambda(lambda x: x, name='last_hidden_state')(all_hidden_states[-1])
    # all_hidden_states = tf.keras.layers.Lambda(lambda x: x, name='all_hidden_states')(all_hidden_states)
    # all_attention_scores = tf.keras.layers.Lambda(
    #     lambda x: x, name='all_attention_scores')(all_attention_scores)
    predictions = tf.keras.layers.Lambda(lambda x: x, name='predictions')(prediction_scores)
    relations = tf.keras.layers.Lambda(lambda x: x, name='relations')(relation_scores)
    print('predictions shape: ', predictions.shape)
    print('relationss shape: ', relations.shape)
    print('last hidden state shape: ', last_hidden_state.shape)

    model = tf.keras.Model(inputs={
        'input_ids': input_ids,
        'input_mask': input_mask,
        'position_ids': position_ids,
        'token_type_ids': token_type_ids,
    }, outputs={
        'predictions': predictions,
        'relations': relations,
        # 'last_hidden_state': last_hidden_state,
        # 'all_hidden_states': all_hidden_states,
        # 'all_attention_scores': all_attention_scores
    })

    model.compile(
        optimizer='adam',
        loss={
            'predictions': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            'relations': tf.keras.losses.BinaryCrossentropy(from_logits=False)
        },
        metrics={
            'predictions': ['acc'],
            'relations': ['acc']
        })
    return model
