import os

import tensorflow as tf
from transformers_keras.modeling_albert import Albert
from transformers_keras.modeling_bert import Bert

BASE_DIR = os.environ.get('BASE_DIR')
if not BASE_DIR:
    BASE_DIR = '/home/zhouyang.lzy/pretrain-models'
os.environ.update({'CUDA_VISIBLE_DEVICES': '-1'})


class LoadPretrainedModelTest(tf.test.TestCase):

    def _do_predict(self, model):
        input_ids = tf.constant([1, 2, 3, 4, 5, 6, 7, 8], shape=(2, 4))
        # output_1 should be all close to output_2
        _, outputs_1, _, _ = model.predict((input_ids,))
        print(outputs_1)
        _, outputs_2, _, _ = model(inputs=(input_ids,))
        print(outputs_2)

    def test_load_pretrained_bert(self):
        model_paths = [
            'chinese_wwm_ext_L-12_H-768_A-12',
            'chinese_L-12_H-768_A-12',
            'chinese_roberta_wwm_ext_L-12_H-768_A-12',
            'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16'
        ]
        for p in model_paths:
            model = Bert.from_pretrained(os.path.join(BASE_DIR, p))
            model.summary()
            self._do_predict(model)

    def test_load_pretrained_albert(self):
        model_paths = [
            'albert_base_zh', 'albert_large_zh', 'albert_xlarge_zh'
        ]
        for p in model_paths:
            model = Albert.from_pretrained(os.path.join(BASE_DIR, p))
            model.summary()
            self._do_predict(model)

    def test_bert_classify(self):

        def _build_bert_model(trainable=True):
            input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids')
            segment_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='segment_ids')

            bert = Bert.from_pretrained(os.path.join(BASE_DIR, 'chinese_roberta_wwm_ext_L-12_H-768_A-12'))
            bert.trainable = trainable

            sequence_output, pooled_output = bert(inputs=(input_ids, segment_ids))
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

            albert = Albert.from_pretrained(os.path.join(BASE_DIR, 'albert_base_zh'))
            albert.trainable = trainable

            sequence_output, pooled_output = albert(inputs=(input_ids, segment_ids))
            outputs = tf.keras.layers.Dense(2, name='output')(pooled_output)
            model = tf.keras.Model(inputs=[input_ids, segment_ids], outputs=outputs)
            model.compile(loss='binary_cross_entropy', optimizer='adam')
            return model

        for trainable in [True, False]:
            model = _build_albert_model(trainable)
            model.summary()


if __name__ == "__main__":
    tf.test.main()
