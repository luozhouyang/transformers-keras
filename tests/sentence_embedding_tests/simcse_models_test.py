import os
import unittest

import tensorflow as tf
from transformers_keras.sentence_embedding.simcse_models import (
    HardNegativeSimCSE, SupervisedSimCSE, UnsupervisedSimCSE)


class SimCSEModelsTest(unittest.TestCase):

    def test_build_simcse_models(self):
        m1 = UnsupervisedSimCSE()
        m1.summary()

        m2 = SupervisedSimCSE()
        m2.summary()

        m3 = HardNegativeSimCSE()
        m3.summary()

    def test_simcse_models_load_pretrained(self):

        def _run_model(model, name):
            model.summary()
            p = os.path.join('models/simcse-{}/1'.format(name))
            model.save(p, include_optimizer=False)

            loaded = tf.saved_model.load(p)
            model = loaded.signatures['serving_default']
            print(model.structured_input_signature)
            print(model.structured_outputs)

        bert_path = os.path.join(os.environ['CHINESE_BERT_PATH'], 'chinese_roberta_wwm_ext_L-12_H-768_A-12')
        m1 = UnsupervisedSimCSE.from_pretrained(bert_path)
        _run_model(m1, 'unsup')

        m2 = SupervisedSimCSE.from_pretrained(bert_path)
        _run_model(m2, 'sup')

        m3 = HardNegativeSimCSE.from_pretrained(bert_path)
        _run_model(m3, 'hardneg')


if __name__ == "__main__":
    unittest.main()
