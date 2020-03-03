import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils


def masked_sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1):
    """Loss for masked language model.

    Args:
        y_true: Tensor, shape (batch_size, time_steps)
        y_pred: Tensor, shape (batch_size, tiem_steps, vocab_size)

    Returns:
        loss: Scalar.
    """
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=from_logits, axis=axis)  # shape (batch_size, time_steps)
    # 0 -> padding token id, masking padding tokens
    mask = tf.logical_not(tf.equal(y_true, 0))  # shape (batch_size, time_steps)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask  # shape (batch_size, time_steps)
    return loss  # do not reduce here!


class MaskedSparseCategoricalCrossentropy(LossFunctionWrapper):
    """Sparse categorical crossentropy for language models, whose labels are padding with special id(e.g 0)."""

    def __init__(self,
                 from_logits=False,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='masked_sparse_categorical_crossentropy'):
        super(MaskedSparseCategoricalCrossentropy, self).__init__(
            masked_sparse_categorical_crossentropy,
            name=name,
            reduction=reduction,
            from_logits=from_logits
        )
