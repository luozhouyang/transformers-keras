import tensorflow as tf

from .losses import MaskedSparseCategoricalCrossentropy
from .metrics import MaskedSparseCategoricalAccuracy
from .modeling_bert import *


def build_model(config):
    input_ids = tf.keras.layers.Input(shape=(config.max_sequence_length,), dtype=tf.int32, name='input_ids')
    input_mask = tf.keras.layers.Input(shape=(config.max_sequence_length,), dtype=tf.int32, name='input_mask')
    segment_ids = tf.keras.layers.Input(shape=(config.max_sequence_length,), dtype=tf.int32, name='segment_ids')

    outputs = Bert4PreTraining(config, name='bert')(inputs=(input_ids, segment_ids, input_mask))

    predictions = tf.keras.layers.Lambda(lambda x: x, name='predictions')(outputs[0])
    relations = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x), name='relations')(outputs[1])
    attentions = tf.keras.layers.Lambda(lambda x: x, name='attentions')(outputs[2])

    model = tf.keras.Model(inputs=[input_ids, segment_ids, input_mask], outputs=[predictions, relations, attentions])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-8, clipnorm=1.0),
        loss={
            'predictions': MaskedSparseCategoricalCrossentropy(mask_id=0, from_logits=True),
            'relations': tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        },
        metrics={
            'predictions': [
                MaskedSparseCategoricalAccuracy(mask_id=0, from_logits=False),
            ],
            'relations': [
                tf.keras.metrics.CategoricalAccuracy(),
            ]
        })
    return model


if __name__ == "__main__":
    config = BertConfig()
    model = build_model(config)
    model.summary()
    tf.keras.utils.plot_model(model, 'model.png', show_shapes=True, expand_nested=True)
