import tensorflow as tf
import os
import tensorflow_datasets as tfds

# Download the file
path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"


class Runner:

    def __init__(self, params):
        self.params = params

    def train(self):
        pass

    def eval(self):
        pass

    def predict(self):
        pass

    def export(self):
        pass


if __name__ == '__main__':
    pass
