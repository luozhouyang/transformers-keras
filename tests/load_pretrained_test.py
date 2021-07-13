import json
import os

import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
from transformers_keras.modeling_albert import Albert
from transformers_keras.modeling_bert import Bert

BASE_DIR = os.environ.get('PRETRAINED_MODE_PATH', None)


class LoadPretrainedModelTest(tf.test.TestCase):

    def _build_bert_inputs(self):
        vocab_path = os.path.join(BASE_DIR, 'bert_uncased_L-6_H-768_A-12', 'vocab.txt')
        tokenizer = BertWordPieceTokenizer(vocab_path)
        encoding = tokenizer.encode('我爱NLP')
        input_ids = tf.constant([encoding.ids], dtype=tf.int32, shape=(1, len(encoding.ids)))
        segment_ids = tf.constant([encoding.type_ids], dtype=tf.int32, shape=(1, len(encoding.type_ids)))
        attention_mask = tf.constant([encoding.attention_mask], dtype=tf.int32, shape=(1, len(encoding.attention_mask)))
        return input_ids, segment_ids, attention_mask

    def _build_model_from_transformers(self):
        from transformers import TFBertModel
        model = TFBertModel.from_pretrained(
            os.path.join(BASE_DIR, 'bert_uncased_L-6_H-768_A-12-pytorch'), from_pt=True)
        return model

    def _build_model(self):
        model = Bert.from_pretrained(
            os.path.join(BASE_DIR, 'bert_uncased_L-6_H-768_A-12'),
            return_states=True, verbose=False)
        return model

    def test_bert_comprare_with_transformers(self):
        model = self._build_model()
        transformer_model = self._build_model_from_transformers()

        input_ids, segment_ids, attention_mask = self._build_bert_inputs()

        def _comprare_embedding_output():
            a = model.bert_embedding(input_ids, segment_ids)
            embedding = transformer_model.get_layer('bert').embeddings
            b = embedding(input_ids=input_ids, token_type_ids=segment_ids, position_ids=None, training=False)
            self.assertAllClose(a, b)

        _comprare_embedding_output()

        def _compare_final_outputs():
            a_sequence_output, a_pooled_output, _ = model(inputs=[input_ids, segment_ids, attention_mask])
            b_sequence_output, b_pooled_output = transformer_model(
                input_ids=input_ids, token_type_ids=segment_ids, attention_mask=attention_mask, return_dict=False)
            # b_sequence_output, b_pooled_output = b_outputs[0], b_outputs[1]
            self.assertAllClose(a_sequence_output, b_sequence_output, rtol=0.5, atol=1e-4)
            self.assertAllClose(a_pooled_output, b_pooled_output, rtol=0.02, atol=1e-4)

        _compare_final_outputs()

    def _build_albert_from_transformers(self):
        from transformers import TFAlbertModel
        model = TFAlbertModel.from_pretrained(os.path.join(BASE_DIR, 'albert_base_zh_pytorch'), from_pt=True)
        return model

    def test_albert_compare_with_transformers(self):
        model = Albert.from_pretrained(os.path.join(BASE_DIR, 'albert_base_zh'))
        transformer_model = self._build_albert_from_transformers()

        input_ids, segment_ids, attention_mask = self._build_bert_inputs()

        def _compare_embedding_output():
            a_output = model.embedding(input_ids, segment_ids)
            b_output = transformer_model.get_layer('albert').embeddings(
                input_ids=input_ids, token_type_ids=segment_ids, position_ids=None, training=False)
            self.assertAllClose(a_output, b_output)

        def _compare_mapping_in_output():
            a_layer = model.encoder.embedding_mapping
            b_layer = transformer_model.get_layer('albert').encoder.embedding_hidden_mapping_in
            a_output = a_layer(model.embedding(input_ids, segment_ids))
            b_output = b_layer(transformer_model.get_layer('albert').embeddings(
                input_ids=input_ids, token_type_ids=segment_ids, position_ids=None, training=False))
            self.assertAllClose(a_output, b_output)

        def _compare_final_outputs():
            a_sequence_output, a_pooled_output = model(input_ids, segment_ids)
            b_sequence_output, b_pooled_output = transformer_model(
                input_ids=input_ids, token_type_ids=segment_ids, attention_mask=attention_mask, return_dict=False)
            self.assertAllClose(a_sequence_output, b_sequence_output, rtol=0.05, atol=1e-5)
            self.assertAllClose(a_pooled_output, b_pooled_output, rtol=0.002, atol=1e-5)

        _compare_embedding_output()
        _compare_mapping_in_output()
        _compare_final_outputs()

    def _do_predict(self, model):
        input_ids = tf.constant([1, 2, 3, 4, 5, 6, 7, 8], shape=(2, 4))
        # output_1 should be all close to output_2
        _, outputs_1 = model(inputs=[input_ids], training=False)
        print(outputs_1)
        # _, outputs_2 = model(input_ids, None, None)
        # print(outputs_2)

    def test_load_pretrained_bert(self):
        model_paths = [
            'chinese_wwm_ext_L-12_H-768_A-12',
            'chinese_L-12_H-768_A-12',
            'chinese_roberta_wwm_ext_L-12_H-768_A-12',
            'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16'
        ]
        for p in model_paths:
            model = Bert.from_pretrained(os.path.join(BASE_DIR, p), verbose=True)
            model.summary()
            self._do_predict(model)

        # skip weights
        model = Bert.from_pretrained(
            os.path.join(BASE_DIR, model_paths[0]),
            skip_token_embedding=True,
            skip_pooler=True,
            verbose=False)
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

        model = Albert.from_pretrained(
            os.path.join(BASE_DIR, model_paths[0]),
            skip_token_embedding=True,
            skip_pooler=True,)
        model.summary()
        self._do_predict(model)

    def test_bert_classify(self):

        def _build_bert_model(trainable=True):
            input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids')
            segment_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='segment_ids')

            bert = Bert.from_pretrained(os.path.join(BASE_DIR, 'chinese_roberta_wwm_ext_L-12_H-768_A-12'))
            bert.trainable = trainable

            sequence_output, pooled_output = bert(inputs=[input_ids, segment_ids])
            outputs = tf.keras.layers.Dense(2, name='output')(pooled_output)
            model = tf.keras.Model(inputs=[input_ids, segment_ids], outputs=outputs)
            model.compile(loss='binary_cross_entropy', optimizer='adam')
            return model

        for trainable in [True]:
            model = _build_bert_model(trainable)
            model.summary()

            model.save('models/export-bert-classify/1', include_optimizer=False)

    def test_albert_classify(self):

        def _build_albert_model(trainable=True):
            input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids')
            segment_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='segment_ids')

            albert = Albert.from_pretrained(os.path.join(BASE_DIR, 'albert_base_zh'))
            albert.trainable = trainable

            sequence_output, pooled_output = albert(input_ids, segment_ids, None)
            outputs = tf.keras.layers.Dense(2, name='output')(pooled_output)
            model = tf.keras.Model(inputs=[input_ids, segment_ids], outputs=outputs)
            model.compile(loss='binary_cross_entropy', optimizer='adam')
            return model

        for trainable in [True, False]:
            model = _build_albert_model(trainable)
            model.summary()


if __name__ == "__main__":
    tf.test.main()
