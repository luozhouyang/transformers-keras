import tensorflow as tf


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
