import tensorflow as tf


def masked_sparse_categorical_accuracy(y_true, y_pred):
    """Accuracy for language models.

    Args:
        y_true: Tensor, shape (batch_size, time_steps)
        y_pred: Tensor, shape (batch_size, time_steps, vocab_size)

    Returns:
        masked_acc: Scalar
    """
    acc = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)  # shape (batch_size, time_steps)
    # 0 -> padding token's id
    mask = tf.logical_not(tf.equal(y_true, 0))
    mask = tf.cast(mask, dtype=acc.dtype)  # shape (batch_size, time_steps)
    numerator = tf.reduce_sum(acc * mask)  # total acc of valid positions
    denominator = tf.reduce_sum(mask) + 1e-5  # total num of valid positions
    masked_acc = numerator / denominator  # final acc of valid tokens
    return masked_acc
