import tensorflow as tf

from transformers_keras.transformer.custom_lr_schedule import CustomLearningRateSchedule


class CustomLearningRateScheduleTest(tf.test.TestCase):

    def testCustomLearningRateSchedule(self):
        temp_learning_rate_schedule = CustomLearningRateSchedule(512)

        # test will failed!
        # use jupyter notebook to show image
        # plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
        # plt.ylabel("Learning Rate")
        # plt.xlabel("Train Step")


if __name__ == '__main__':
    tf.test.main()
