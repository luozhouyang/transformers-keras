import tensorflow as tf
from tensorflow.python.ops import math_ops


def masked_sparse_categorical_crossentropy(y_true, y_pred):
    """Sparse categorical crossentropy with masking support for language models.

    Args:
        y_true: Tensor, shape (batch_size, time_steps)
        y_pred: Tensor, shape (batch_size, time_steps, vocab_size), outputs of softmax

    Returns:
        A loss scalar.
    """
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1)
    # masking positions where value equals 0 (usually padding token's id)
    mask = tf.logical_not(tf.equal(y_true, 0))  # shape (batch_size, time_steps)
    mask = tf.cast(mask, dtype=loss.dtype)
    numerator = tf.reduce_sum(loss * mask)  # total loss of valid positions
    denominator = tf.reduce_sum(mask) + 1e-6  # total number of valid position
    loss = numerator / denominator  # average loss
    return loss


class MaskedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    """Sparse categorical crossentropy with masking support."""

    def __init__(self, mask_id=0, from_logits=False, axis=-1, **kwargs):
        """Init.

        Args:
            mask_id: Python integer, value from `y_true` which needs to be masking.
            from_logits: Python boolean, `y_pred` is logits or not.
            axis: Python integer, which dimension to compute crossentropy
            epsilone: Python float, avoid divid by zero error
        """
        super().__init__(reduction='none', **kwargs)  # we do reduction mannually
        self.mask_id = mask_id
        self.from_logits = from_logits
        self.axis = axis

    def call(self, y_true, y_pred):
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=self.from_logits, axis=self.axis)
        # masking positions where value equals `mask_id` (usually padding token's id)
        mask = tf.logical_not(tf.equal(y_true, self.mask_id))  # shape (batch_size, time_steps)
        mask = tf.cast(mask, dtype=loss.dtype)
        numerator = tf.reduce_sum(loss * mask)  # total loss of valid positions
        denominator = tf.reduce_sum(mask)  # total number of valid position
        loss = math_ops.div_no_nan(numerator, denominator)  # average loss over valid positions
        return loss
