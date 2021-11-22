import logging
import os
import unittest

import tensorflow as tf
from smile_datasets import DatapipeForMaksedLanguageModel
from tokenizers import BertWordPieceTokenizer
from transformers_keras.lm.mlm import BertForMaskedLanguageModel

BERT_PATH = os.path.join(os.environ["CHINESE_BERT_PATH"], "chinese_roberta_wwm_ext_L-3_H-768_A-12")


class MaskedLanguageModelTest(tf.test.TestCase):
    """Tests for masked language model."""

    def test_bert_for_mlm(self):
        m = BertForMaskedLanguageModel.from_pretrained(BERT_PATH, with_mlm=True)
        m.summary()

    def test_bert_for_mlm_train(self):
        d = DatapipeForMaksedLanguageModel.from_tfrecord_files(
            input_files=["testdata/mlm.tfrecord"], batch_size=2, repeat=100
        )
        m = BertForMaskedLanguageModel.from_pretrained(BERT_PATH, with_mlm=True)
        m.summary()
        m.compile(optimizer="adam")
        m.fit(d)

        m.save("models/bert-for-mlm/1")

    def test_bert_for_mlm_load(self):
        m = tf.saved_model.load("models/bert-for-mlm/1")
        m = m.signatures["serving_default"]
        print(m.structured_input_signature)
        print(m.structured_outputs)

    def _build_model_inputs(self):
        vocab_path = os.path.join(os.environ["GOOGLE_BERT_PATH"], "uncased_L-6_H-768_A-12", "vocab.txt")
        tokenizer = BertWordPieceTokenizer(vocab_path)
        encoding = tokenizer.encode("I love NLP, Neural [MASK] Processing is amazing!")
        logging.info("   ids: %s", encoding.ids)
        logging.info("tokens: %s", encoding.tokens)
        input_ids = tf.constant([encoding.ids], dtype=tf.int32, shape=(1, len(encoding.ids)))
        segment_ids = tf.constant([encoding.type_ids], dtype=tf.int32, shape=(1, len(encoding.type_ids)))
        attention_mask = tf.constant([encoding.attention_mask], dtype=tf.int32, shape=(1, len(encoding.attention_mask)))
        return input_ids, segment_ids, attention_mask

    def _build_masked_lm_from_huggingface(self):
        from transformers import TFBertForMaskedLM

        model = TFBertForMaskedLM.from_pretrained(
            os.path.join(os.environ["PYTORCH_MODEL_PATH"], "bert_uncased_L-6_H-768_A-12-pytorch"),
            from_pt=True,
        )
        return model

    def _build_masked_lm(self):
        model = BertForMaskedLanguageModel.from_pretrained(
            os.path.join(os.environ["GOOGLE_BERT_PATH"], "uncased_L-6_H-768_A-12"),
            with_mlm=True,
            override_params={"epsilon": 1e-12},
        )
        return model

    def test_compare_with_huggingface(self):
        # TODO
        my_model = self._build_masked_lm()
        hg_model = self._build_masked_lm_from_huggingface()

        input_ids, segment_ids, attention_mask = self._build_model_inputs()

        def _compare_embedding_outputs():
            a = my_model.bert_model.bert_embedding(input_ids, segment_ids)
            embedding = hg_model.get_layer("bert").embeddings
            b = embedding(input_ids=input_ids, token_type_ids=segment_ids, position_ids=None, training=False)
            self.assertAllClose(a, b)

        def _compare_hidden_states():
            a_sequence, a_pooled, a_hidden, a_attn = my_model.bert_model(
                input_ids, segment_ids, attention_mask, training=False
            )
            b_sequence, b_pooled, b_hidden, b_attn = hg_model.get_layer("bert")(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=attention_mask,
                return_dict=False,
                output_attentions=True,
                output_hidden_states=True,
            )
            b_attn = tf.transpose(b_attn, [1, 0, 2, 3, 4])
            self.assertAllClose(a_attn, b_attn, rtol=7e-4, atol=9e-5)
            self.assertAllClose(a_sequence, b_sequence, rtol=3e-1, atol=7e-5)

        def _compare_final_outputs():
            a_pred = my_model(inputs=[input_ids, segment_ids, attention_mask])
            outputs = hg_model(
                input_ids=input_ids, token_type_ids=segment_ids, attention_mask=attention_mask, return_dict=True
            )
            b_pred = outputs["logits"]
            self.assertAllClose(a_pred, b_pred, rtol=7e-1, atol=4e-4)

        _compare_embedding_outputs()
        logging.info("Passed comparing embedding outputs...")
        _compare_hidden_states()
        logging.info("Passed comparing hidden states...")
        _compare_final_outputs()
        logging.info("Passed comparing final outputs...")


if __name__ == "__main__":
    unittest.main()
