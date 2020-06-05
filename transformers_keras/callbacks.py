import logging
import os
import time

import tensorflow as tf


class TransformerLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule for Transformer."""

    def __init__(self, depth, warmup_steps=4000):
        """
        Args:
            depth: Python integer, the model's hidden size
            warmup_steps: Python integer, steps to warmup learning rate
        """
        self.depth = depth
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(tf.cast(self.depth, tf.float32)) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {
            'depth': self.depth,
            'warmup_steps': self.warmup_steps
        }
        return config


class SavedModelExporter(tf.keras.callbacks.Callback):
    """Export saved model every n epochs or every n steps."""

    def __init__(self, export_dir, every_epoch=1, every_steps=None):
        super(SavedModelExporter, self).__init__()
        self.export_dir = export_dir
        self.every_epoch = every_epoch
        self.every_steps = every_steps
        self.epoch = 0

    def on_train_batch_end(self, batch, logs=None):
        if self.every_steps is None:
            return
        if self.every_steps == 0:
            raise ValueError('`every_steps` can not be set to 0.')
        if (batch + 1) % self.every_steps == 0:
            millis = int(round(time.time() * 1000))
            p = os.path.join(self.export_dir, 'epoch-{}-step-{}-{}'.format(self.epoch, batch + 1, millis))
            if not os.path.exists(p):
                os.makedirs(p)
            self.model.save(p, include_optimizer=False, save_format='tf')
            logging.info('Export model in SavedModel format to {} at batch {}.'.format(p, batch + 1))

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        if self.every_epoch is None:
            return
        if self.every_epoch == 0:
            raise ValueError('`every_epoch` can not be set to 0.')
        if (epoch + 1) % self.every_epoch == 0:
            millis = int(round(time.time() * 1000))
            p = os.path.join(self.export_dir, 'epoch-{}-{}'.format(epoch, millis))
            if not os.path.exists(p):
                os.makedirs(p)
            self.model.save(p, include_optimizer=False, save_format='tf')
            logging.info('Export model in SavedModel format to {} at epoch {}.'.format(p, epoch))
