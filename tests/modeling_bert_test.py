import unittest

import numpy as np
import tensorflow as tf
from transformers_keras.modeling_bert import *


class ModelingBertTest(tf.test.TestCase):

    def testBertEmbeddings(self):
        embedding_layer = BertEmbedding(vocab_size=16, max_positions=40)
        inputs = (
            tf.constant([[0, 2, 3, 4, 5, 1]]),
            # tf.constant([[0, 1, 2, 3, 4, 5]]),
            tf.constant([[0, 0, 0, 1, 1, 1]]),
        )
        embeddings = embedding_layer(inputs, mode='embedding', training=True)
        self.assertAllEqual([1, 6, 768], embeddings.shape)

        inputs2 = tf.random.uniform(shape=(1, 768))
        outputs = embedding_layer(inputs2, mode='linear')
        self.assertAllEqual([1, 16], outputs.shape)

    def testBertIntermediate(self):
        inter = BertIntermediate()
        inputs = tf.random.uniform(shape=(2, 16, 768))
        outputs = inter(inputs)
        self.assertAllEqual([2, 16, 3072], outputs.shape)

    def testBertEncoderLayer(self):
        encoder = BertEncoderLayer()
        hidden_states = tf.random.uniform((2, 10, 768))
        attention_mask = None
        outputs, attention_weights = encoder(hidden_states, attention_mask, training=True)

        self.assertAllEqual([2, 10, 768], outputs.shape)
        self.assertAllEqual([2, 8, 10, 10], attention_weights.shape)

    def testBertEncoder(self):
        encoder = BertEncoder(num_layers=2)

        hidden_states = tf.random.uniform((2, 10, 768))
        attention_mask = None
        outputs, all_hidden_states, all_attention_scores = encoder(hidden_states, attention_mask, training=True)

        self.assertAllEqual([2, 10, 768], outputs.shape)

        self.assertAllEqual(2, len(all_hidden_states))
        for state in all_hidden_states:
            self.assertAllEqual([2, 10, 768], state.shape)

        self.assertAllEqual(2, len(all_attention_scores))
        for attention in all_attention_scores:
            self.assertAllEqual([2, 8, 10, 10], attention.shape)

    def testBertPooler(self):
        pooler = BertPooler()
        inputs = tf.random.uniform(shape=(2, 16, 768))
        outputs = pooler(inputs)
        self.assertAllEqual([2, 768], outputs.shape)

    def _build_bert_inputs(self):
        input_ids = tf.constant(
            [1, 2, 3, 4, 5, 6, 7, 5, 3, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5, 6, 6, 6, 7, 7, 8, 0, 0, 0, 0, 0, 0],
            shape=(2, 16),
            dtype=np.int32)  # input_ids
        token_type_ids = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int64)  # token_type_ids,
        input_mask = tf.constant(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]], dtype=np.float32)  # input_mask
        return input_ids, token_type_ids, input_mask

    def _check_bert_outputs(self, return_states=False, return_attention_weights=False):
        model = Bert(
            vocab_size=100,
            num_layers=2,
            return_states=return_states,
            return_attention_weights=return_attention_weights)
        input_ids, segment_ids, attn_mask = self._build_bert_inputs()
        outputs = model(input_ids, segment_ids, attn_mask)
        sequence_outputs, pooled_outputs = outputs[0], outputs[1]
        self.assertAllEqual([2, 16, 768], sequence_outputs.shape)
        self.assertAllEqual([2, 768], pooled_outputs.shape)

        all_states, all_attn_weights = None, None
        if return_states and return_attention_weights:
            self.assertEqual(4, len(outputs))
            all_states, all_attn_weights = outputs[2], outputs[3]
        elif return_states and not return_attention_weights:
            self.assertEqual(3, len(outputs))
            all_states = outputs[2]
        elif not return_states and return_attention_weights:
            self.assertEqual(3, len(outputs))
            all_attn_weights = outputs[2]
        else:
            self.assertEqual(2, len(outputs))

        if all_states is not None:
            self.assertEqual(2, len(all_states))
            for state in all_states:
                self.assertAllEqual([2, 16, 768], state.shape)

        if all_attn_weights is not None:
            self.assertEqual(2, len(all_attn_weights))
            for attention in all_attn_weights:
                self.assertAllEqual([2, 8, 16, 16], attention.shape)

    def testBert(self):
        self._check_bert_outputs(return_states=True, return_attention_weights=True)
        self._check_bert_outputs(return_states=True, return_attention_weights=False)
        self._check_bert_outputs(return_states=False, return_attention_weights=True)
        self._check_bert_outputs(return_states=False, return_attention_weights=False)

    def testBertMLMHead(self):
        embedding = BertEmbedding(vocab_size=100)
        mlm = BertMLMHead(vocab_size=100, embedding=embedding)

        inputs = tf.random.uniform(shape=(2, 16, 768))
        outputs = mlm(inputs)
        self.assertAllEqual([2, 16, mlm.vocab_size], outputs.shape)

    def testBertNSPHead(self):
        nsp = BertNSPHead()
        inputs = tf.random.uniform(shape=(2, 768))
        outputs = nsp(inputs)
        self.assertAllEqual([2, 2], outputs.shape)

    def test_bert_config(self):
        model = Bert(
            vocab_size=100,
            num_layers=2,
            return_states=True,
            return_attention_weights=True)
        config = model.get_config()
        print(config)


if __name__ == "__main__":
    unittest.main()
