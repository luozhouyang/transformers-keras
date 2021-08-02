import os
import unittest

import tensorflow as tf

from transformers_keras.token_classification.crf_models import (
    AlertCRFForTokenClassification,
    BertCRFForTokenClassification,
    CRFModel,
)


class CRFModelsTest(unittest.TestCase):
    """CRF models test."""

    def test_crf_model(self):
        sequence_input = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="sequence_input")
        outputs = tf.keras.layers.Embedding(21128, 128)(sequence_input)
        outputs = tf.keras.layers.Dense(256)(outputs)
        base = tf.keras.Model(inputs=sequence_input, outputs=outputs)
        m = CRFModel(base, 5)
        m.summary()
        m.save("models/crf-model/1")

    def test_bert_crf_for_token_classification(self):
        m = BertCRFForTokenClassification(4)
        m.summary()
        m.save("models/bert-crf-for-token-classification/1", signatures=m.forward)

        bert_path = os.path.join(os.environ["CHINESE_BERT_PATH"], "chinese_roberta_wwm_ext_L-12_H-768_A-12")
        m = BertCRFForTokenClassification.from_pretrained(bert_path, override_params={"num_labels": 4})
        m.summary()
        m.save("models/bert-crf-for-token-classification/2", signatures=m.forward)

    def test_load_bert_crf_for_token_classification(self):
        # load from saved model
        m = tf.saved_model.load("models/bert-crf-for-token-classification/2")
        model = m.signatures["serving_default"]
        print(model.structured_input_signature)
        print(model.structured_outputs)

    def test_albert_crf_for_token_classification(self):
        m = AlertCRFForTokenClassification(4)
        m.summary()
        m.save("models/albert-crf-for-token-classification/1", signatures=m.forward)

        albert_path = os.path.join(os.environ["GOOGLE_ALBERT_PATH"], "albert-base-zh")
        m = AlertCRFForTokenClassification.from_pretrained(albert_path, override_params={"num_labels": 4})
        m.summary()
        m.save("models/albert-crf-for-token-classification/2", signatures=m.forward)

    def test_load_albert_crf_for_token_classification(self):
        # load from saved model
        m = tf.saved_model.load("models/albert-crf-for-token-classification/2")
        model = m.signatures["serving_default"]
        print(model.structured_input_signature)
        print(model.structured_outputs)

    def test_bert_predict(self):
        m = BertCRFForTokenClassification(4)
        input_ids, segment_ids, attention_mask = m.dummy_inputs()
        dataset = tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensor_slices([input_ids]),
                tf.data.Dataset.from_tensor_slices([segment_ids]),
                tf.data.Dataset.from_tensor_slices([attention_mask]),
            )
        )
        dataset = dataset.map(lambda a, b, c: ({"input_ids": a, "segment_ids": b, "attention_mask": c}, None))
        pred = m.predict(dataset)
        print("\npred: ", pred)


if __name__ == "__main__":
    unittest.main()
