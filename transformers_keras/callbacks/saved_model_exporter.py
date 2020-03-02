import os
import time

import tensorflow as tf


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
