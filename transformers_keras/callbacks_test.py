
import tensorflow as tf

from .callbacks import ModelStepCheckpoint, SavedModelExporter


class SavedModelExporterTest(tf.test.TestCase):

    def buildModel(self):
        x = tf.keras.layers.Input(shape=(2,), name='x', dtype=tf.int32)
        h = tf.keras.layers.Dense(16)(x)
        h = tf.keras.layers.Dense(2)(h)
        o = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x), name='o')(h)
        m = tf.keras.Model(inputs=x, outputs=o)
        opt = tf.keras.optimizers.Adam()
        m.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
        return m, opt

    def testExportSavedModel(self):
        model, _ = self.buildModel()
        model.build(input_shape=[None, 2])
        model.summary()

        x = tf.data.Dataset.from_tensor_slices(tf.constant([1, 2], dtype=tf.int32, shape=(1, 2)))
        y = tf.data.Dataset.from_tensor_slices(tf.constant([0, 1], dtype=tf.int32, shape=(1, 2)))
        dataset = tf.data.Dataset.zip((x, y))
        dataset = dataset.repeat(100).shuffle(100).batch(2, drop_remainder=True)

        model.fit(
            dataset,
            validation_data=dataset,
            epochs=10,
            callbacks=[
                SavedModelExporter(model, '/tmp/ckpt/export', 1, 40)
            ]
        )


if __name__ == "__main__":
    tf.test.main()
