import os

import tensorflow as tf

from transformers_keras.modeling_bert import Bert


BASE_DIR = os.environ.get('BASE_DIR')


class LoadPretrainedModelTest(tf.test.TestCase):

    def test_load_pretrained_bert(self):
        model1 = Bert.from_pretrained(os.path.join(BASE_DIR, 'chinese_wwm_ext_L-12_H-768_A-12'))
        model1.summary()
        
        model2 = Bert.from_pretrained(os.path.join(BASE_DIR, 'chinese_L-12_H-768_A-12'))
        model2.summary()

        model3 = Bert.from_pretrained(os.path.join(BASE_DIR, 'chinese_roberta_wwm_ext_L-12_H-768_A-12'))
        model3.summary()

        model4 = Bert.from_pretrained(os.path.join(BASE_DIR, 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16'))
        model4.summary()

if __name__ == "__main__":
    tf.test.main()

