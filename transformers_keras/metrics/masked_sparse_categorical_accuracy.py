import tensorflow as tf
from tensorflow.python.keras.metrics import MeanMetricWrapper
from tensorflow.python.ops import math_ops


class MaskedSparseCategoricalAccuracy(tf.keras.metrics.Metric):

    def __init__(self, name='masked_sparse_categorical_accuracy', from_logits=False, mask_id=0, **kwargs):
        super(MaskedSparseCategoricalAccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, from_logits=False, mask_id=0):
        """Update state of one training step.

        Args:
            y_true: Tensor, with shape (batch_size, time_steps)
            y_pred: Tensor, with shape (batch_size, time_steps, vocab_size)
        """
        if from_logits:
            y_pred = tf.nn.softmax(y_pred)
        acc = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)  # shape (batch_size, time_steps)
        # 0 -> padding token's id
        mask = tf.logical_not(tf.equal(y_true, mask_id))
        mask = tf.cast(mask, dtype=acc.dtype)  # shape (batch_size, time_steps)

        values = tf.reduce_sum(acc*mask)
        num_values = tf.reduce_sum(mask)
        self.total.assign_add(values)
        self.count.assign_add(num_values)

    def result(self):
        return math_ops.div_no_nan(self.total, self.count)
