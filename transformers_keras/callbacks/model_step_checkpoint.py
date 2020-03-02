import tensorflow as tf


class ModelStepCheckpoint(tf.keras.callbacks.Callback):
    """Save model in Checkpoint format every n steps."""

    def __init__(self, model, ckpt_dir, every_steps=10000, max_keep_ckpt=5, restore_on_train_begin=True):
        super(ModelStepCheckpoint, self).__init__()
        self.model = model
        self.ckpt_dir = ckpt_dir
        self.every_steps = every_steps
        self.max_keep_ckpt = max_keep_ckpt
        self.restore_on_train_begin = restore_on_train_begin
        self.global_step = tf.Variable(0, dtype=tf.int32)
        self.ckpt = tf.train.Checkpoint(model=model, step=self.global_step)
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, self.ckpt_dir, max_to_keep=self.max_keep_ckpt
        )
        self.epoch = 0

    def on_train_begin(self, logs=None):
        if self.restore_on_train_begin:
            try:
                latest_ckpt = self.ckpt_manager.latest_checkpoint
                if not latest_ckpt:
                    print('No latest checkpoint found. Skip to restore.')
                    return
                status = self.ckpt.restore(latest_ckpt)
                print('Restore lastest checkpoint from {}. Global step is {}.'.format(
                    latest_ckpt, self.global_step.numpy()))
            except Exception as e:
                print('Restore latest checkpoint failed. Exception is {}'.format(e))

    def on_train_end(self, logs=None):
        t = int(self.global_step.numpy())
        p = self.ckpt_manager.save(t)
        print('End training. Global step is {}. Model saved in checkpoint format to {}.'.format(self.global_step.numpy(), p))

    def on_train_batch_end(self, batch, logs=None):
        self.global_step.assign_add(1)
        if self.every_steps is None:
            return
        t = int(self.global_step.numpy())
        if t % self.every_steps == 0:
            p = self.ckpt_manager.save(t)
            print('\nModel saved in checkpoint format to {} at batch {} of epoch {}.'.format(p, batch + 1, self.epoch))

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
