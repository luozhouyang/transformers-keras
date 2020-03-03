import tensorflow as tf
from tensorflow.python.keras.metrics import MeanMetricWrapper


def masked_sparse_categorical_accuracy(y_true, y_pred):
    """Accuracy for masked language model.

    Args:
        y_true: Tensor, shape (batch_size, time_steps)
        y_pred: Tensor, shape (batch_size, time_steps, vocab_size), outputs of softmax

    Returns:
        acc: Scalar.
    """
    acc = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)  # shape (batch_size, time_steps)
    # 0 -> padding token's id
    mask = tf.logical_not(tf.equal(y_true, 0))
    mask = tf.cast(mask, dtype=acc.dtype)  # shape (batch_size, time_steps)
    acc *= mask
    return acc  # do not do reduction here!


class MaskedSparseCategoricalAccuracy(MeanMetricWrapper):

    def __init__(self, name='masked_sparse_categorical_accuracy', dtype=None):
        super(MaskedSparseCategoricalAccuracy, self).__init__(
            masked_sparse_categorical_accuracy, name, dtype=dtype)
