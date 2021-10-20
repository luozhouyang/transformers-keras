import os
import unittest

import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
from transformers_keras import Bert, BertForMaskedLanguageModel


class MengziAdapterTest(tf.test.TestCase):
    """Mengzi adapter tests"""

    def _check_envs(self):
        if any(os.environ.get(x, None) is None for x in ['MENGZI_MODEL_PATH', 'GOOGLE_BERT_PATH']):
            return False
        try:
            import torch
        except Exception as e:
            return False
        
        return True
        

    def test_load_mengzi(self):
        if not self._check_envs():
            return

        m = Bert.from_pretrained(
            pretrained_model_dir=os.environ["MENGZI_MODEL_PATH"],
            adapter="langboat/mengzi",
        )
        m.summary()

        m = BertForMaskedLanguageModel.from_pretrained(
            pretrained_model_dir=os.environ["MENGZI_MODEL_PATH"],
            adapter="langboat/mengzi",
            with_mlm=True,
        )
        m.summary()

    def _build_bert_inputs(self):
        vocab_path = os.path.join(os.environ["GOOGLE_BERT_PATH"], "uncased_L-6_H-768_A-12", "vocab.txt")
        tokenizer = BertWordPieceTokenizer(vocab_path)
        encoding = tokenizer.encode("我爱NLP")
        input_ids = tf.constant([encoding.ids], dtype=tf.int32, shape=(1, len(encoding.ids)))
        segment_ids = tf.constant([encoding.type_ids], dtype=tf.int32, shape=(1, len(encoding.type_ids)))
        attention_mask = tf.constant([encoding.attention_mask], dtype=tf.int32, shape=(1, len(encoding.attention_mask)))
        return input_ids, segment_ids, attention_mask

    def _build_torch_model(self):
        from transformers import TFBertModel

        model = TFBertModel.from_pretrained(os.environ["MENGZI_MODEL_PATH"], from_pt=True)
        return model

    def _build_keras_model(self):
        model = Bert.from_pretrained(
            pretrained_model_dir=os.environ["MENGZI_MODEL_PATH"],
            adapter="langboat/mengzi",
            override_params={"epsilon": 1e-12}
        )
        return model

    def test_compare_results(self):
        if not self._check_envs():
            return
            
        torch_model = self._build_torch_model()
        keras_model = self._build_keras_model()

        input_ids, segment_ids, attention_mask = self._build_bert_inputs()

        def _comprare_embedding_output():
            a = keras_model.bert_model.bert_embedding(input_ids, segment_ids)
            embedding = torch_model.get_layer("bert").embeddings
            b = embedding(input_ids=input_ids, token_type_ids=segment_ids, position_ids=None, training=False)
            self.assertAllClose(a, b)

        _comprare_embedding_output()

        xw = sorted([w for w in keras_model.trainable_weights], key=lambda x: x.name)
        yw = sorted([w for w in torch_model.trainable_weights], key=lambda x: x.name)
        for x, y in zip(xw, yw):
            # print("{} -> {}".format(x.name, y.name))
            try:
                self.assertAllClose(x.numpy(), y.numpy())
            except Exception:
                print('{} -> {} not close!'.format(x.name, y.name))
                print(x.numpy())
                print()
                print(y.numpy())
                print('=' * 80)
                continue
            
        def _compare_final_outputs():
            a_sequence_output, a_pooled_output = keras_model(
                inputs=[input_ids, segment_ids, attention_mask], training=False
            )
            b_sequence_output, b_pooled_output = torch_model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )
            self.assertAllClose(a_sequence_output, b_sequence_output, atol=2e-4, rtol=2e-2)
            self.assertAllClose(a_pooled_output, b_pooled_output, atol=3e-5, rtol=8e-3)

        _compare_final_outputs()


if __name__ == "__main__":
    unittest.main()
