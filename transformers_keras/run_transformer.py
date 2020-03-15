import tensorflow as tf

from .modeling_transformer import *
from .losses import MaskedSparseCategoricalCrossentropy
from .metrics import MaskedSparseCategoricalAccuracy


def build_model(config):
    x = tf.keras.layers.Input(shape=(config.max_positions,), dtype=tf.int32, name='x')
    y = tf.keras.layers.Input(shape=(config.max_positions,), dtype=tf.int32, name='y')
    transformer = Transformer(config)
    logits, enc_attns, dec_attns_0, dec_attens_1 = transformer(inputs=(x, y))
    probs = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x), name='probs')(logits)

    model = tf.keras.Model(inputs=[x, y], outputs=[probs])

    model.compile(
        optimizer='adam',  # TODO(zhouyang.luo) schedule learning rate
        loss={
            'probs': MaskedSparseCategoricalCrossentropy(mask_id=0, from_logits=False),
        },
        metrics={
            'probs': [
                MaskedSparseCategoricalAccuracy(mask_id=0, from_logits=False),
            ]
        }
    )

    return model


if __name__ == "__main__":
    config = TransformerConfig()
    model = build_model(config)

    model.summary()
    tf.keras.utils.plot_model(model, 'transformer.png', show_shapes=True, expand_nested=False)
