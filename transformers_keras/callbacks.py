import os
import time

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


class SavedModelExporter(tf.keras.callbacks.Callback):
    """Export saved model every n epochs or every n steps."""

    def __init__(self, model, export_dir, every_epoch=1, every_steps=None):
        super(SavedModelExporter, self).__init__()
        self.model = model
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
            print('Export model in SavedModel format to {} at batch {}.'.format(p, batch + 1))

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
            print('Export model in SavedModel format to {} at epoch {}.'.format(p, epoch))


class ModelStepCheckpoint(tf.keras.callbacks.Callback):
    """Save model in Checkpoint format every n steps."""

    def __init__(self, model, ckpt_dir, every_steps=10000, max_keep_ckpt=5):
        super(ModelStepCheckpoint, self).__init__()
        self.model = model
        self.ckpt_dir = ckpt_dir
        self.every_steps = every_steps
        self.max_keep_ckpt = max_keep_ckpt
        self.ckpt = tf.train.Checkpoint(model=model)
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, self.ckpt_dir, max_to_keep=self.max_keep_ckpt
        )
        self.epoch = 0

    def on_train_end(self, logs=None):
        p = self.ckpt_manager.save()
        print('Model saved in checkpoint format to {}.'.format(p))

    def on_train_batch_end(self, batch, logs=None):
        if self.every_steps is None:
            return
        if (batch+1) % self.every_steps == 0:
            p = self.ckpt_manager.save()
            print('\nModel saved in checkpoint format to {} at batch {} of epoch {}.'.format(p, batch + 1, self.epoch))

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
