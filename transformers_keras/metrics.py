import tensorflow as tf
from tensorflow.python.ops import math_ops


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


class MaskedSparseCategoricalAccuracy(tf.keras.metrics.Metric):

    def __init__(self, mask_id=0, from_logits=True, **kwargs):
        super(MaskedSparseCategoricalAccuracy, self).__init__(**kwargs)
        self.mask_id = mask_id
        self.from_logits = from_logits

        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)
        acc = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = tf.cast(acc, self.dtype)
        mask = tf.logical_not(tf.equal(y_true, self.mask_id))
        mask = tf.cast(mask, self.dtype)
        total_values = tf.reduce_sum(acc * mask)  # value of valid positions
        total_nums = tf.reduce_sum(mask)  # number of valid positions
        self.total.assign_add(total_values)
        self.count.assign_add(total_nums)

    def result(self):
        # if tf.equal(tf.cast(0.0, dtype=self.total.dtype), self.total):
        #     return tf.cast(0.0, dtype=self.dtype)
        # if tf.equal(tf.cast(0.0, dtype=self.count.dtype), self.count):
        #     return tf.cast(0.0, dtype=self.dtype)
        # return tf.cast(self.total / self.count, self.dtype)
        return math_ops.div_no_nan(self.total, self.count)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)
