import os

import tensorflow as tf
from transformers_keras.modeling_albert import Albert
from transformers_keras.modeling_bert import Bert

BASE_DIR = os.environ.get('BASE_DIR')
if not BASE_DIR:
    BASE_DIR = '/home/zhouyang.lzy/pretrain-models'
os.environ.update({'CUDA_VISIBLE_DEVICES': '-1'})


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

    def test_load_pretrained_albert(self):
        model1 = Albert.from_pretrained(os.path.join(BASE_DIR, 'albert_base_zh'))
        model1.summary()

        model2 = Albert.from_pretrained(os.path.join(BASE_DIR, 'albert_large_zh'))
        model2.summary()

        model3 = Albert.from_pretrained(os.path.join(BASE_DIR, 'albert_xlarge_zh'))
        model3.summary()

    def test_bert_classify(self):

        def _build_bert_model(trainable=True):
            input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids')
            segment_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='segment_ids')

            bert = Bert.from_pretrained(os.path.join(BASE_DIR, 'chinese_wwm_ext_L-12_H-768_A-12'))
            bert.trainable = trainable

            sequence_output, pooled_output, _, _ = bert(inputs=(input_ids, segment_ids))
            outputs = tf.keras.layers.Dense(2, name='output')(pooled_output)
            model = tf.keras.Model(inputs=[input_ids, segment_ids], outputs=outputs)
            model.compile(loss='binary_cross_entropy', optimizer='adam')
            return model

        for trainable in [True, False]:
            model = _build_bert_model(trainable)
            model.summary()

    def test_albert_classify(self):

        def _build_albert_model(trainable=True):
            input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids')
            segment_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='segment_ids')

            albert = Albert.from_pretrained(os.path.join(BASE_DIR, 'albert_large_zh'))
            albert.trainable = trainable
            
            sequence_output, pooled_output, _, _ = albert(inputs=(input_ids, segment_ids))
            outputs = tf.keras.layers.Dense(2, name='output')(pooled_output)
            model = tf.keras.Model(inputs=[input_ids, segment_ids], outputs=outputs)
            model.compile(loss='binary_cross_entropy', optimizer='adam')
            return model

        for trainable in [True, False]:
            model = _build_albert_model(trainable)
            model.summary()


if __name__ == "__main__":
    tf.test.main()
