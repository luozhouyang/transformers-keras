import tensorflow as tf


class TransformerLearningRate(tf.keras.optimizers.schedules.LearningRateSchdule):
    """Learning rate schedule for Transformer."""

    def __init__(self, depth, warmup_steps=4000):
        """
        Args:
            depth: Python integer, the model's hidden size
            warmup_steps: Python integer, steps to warmup learning rate
        """
        self.depth = tf.cast(depth, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
