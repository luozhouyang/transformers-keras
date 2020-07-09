import numpy as np
import tensorflow as tf

from .modeling_bert import (
    Bert,
    BertEmbedding,
    BertEncoder,
    BertEncoderLayer,
    BertForPretrainingModel,
    BertIntermediate,
    BertMLMHead,
    BertModel,
    BertNSPHead,
    BertPooler,
)


class ModelingBertTest(tf.test.TestCase):

    def testBertEmbeddings(self):
        embedding_layer = BertEmbedding(vocab_size=16, max_positions=40)
        inputs = (
            tf.constant([[0, 2, 3, 4, 5, 1]]),
            # tf.constant([[0, 1, 2, 3, 4, 5]]),
            tf.constant([[0, 0, 0, 1, 1, 1]]),
        )
        embeddings = embedding_layer(inputs, mode='embedding', training=True)
        self.assertAllEqual([1, 6, embedding_layer.hidden_size], embeddings.shape)

        inputs2 = tf.random.uniform(shape=(1, embedding_layer.hidden_size))
        outputs = embedding_layer(inputs2, mode='linear')
        self.assertAllEqual([1, embedding_layer.vocab_size], outputs.shape)

    def testBertIntermediate(self):
        inter = BertIntermediate()
        inputs = tf.random.uniform(shape=(2, 16, 768))
        outputs = inter(inputs)
        self.assertAllEqual([2, 16, inter.intermediate_size], outputs.shape)

    def testBertEncoderLayer(self):
        encoder = BertEncoderLayer()
        hidden_states = tf.random.uniform((2, 10, encoder.hidden_size))
        attention_mask = None
        outputs, attention_weights = encoder(
            inputs=(hidden_states, attention_mask), training=True)

        self.assertAllEqual([2, 10, encoder.hidden_size], outputs.shape)
        self.assertAllEqual([2, encoder.num_attention_heads, 10, 10], attention_weights.shape)

    def testBertEncoder(self):
        encoder = BertEncoder(num_layers=2)

        hidden_states = tf.random.uniform((2, 10, encoder.hidden_size))
        attention_mask = None
        outputs, all_hidden_states, all_attention_scores = encoder(
            inputs=(hidden_states, attention_mask), training=True)

        self.assertAllEqual([2, 10, encoder.hidden_size], outputs.shape)

        self.assertAllEqual(encoder.num_layers, len(all_hidden_states))
        for state in all_hidden_states:
            self.assertAllEqual([2, 10, encoder.hidden_size], state.shape)

        self.assertAllEqual(encoder.num_layers, len(all_attention_scores))
        for attention in all_attention_scores:
            self.assertAllEqual([2, encoder.num_attention_heads, 10, 10], attention.shape)

    def testBertPooler(self):
        pooler = BertPooler()
        inputs = tf.random.uniform(shape=(2, 16, 768))
        outputs = pooler(inputs)
        self.assertAllEqual([2, pooler.hidden_size], outputs.shape)

    def testBert(self):
        model = Bert(vocab_size=100, num_layers=2)
        input_ids = tf.constant(
            [1, 2, 3, 4, 5, 6, 7, 5, 3, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5, 6, 6, 6, 7, 7, 8, 0, 0, 0, 0, 0, 0],
            shape=(2, 16),
            dtype=np.int32)  # input_ids
        token_type_ids = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int64)  # token_type_ids,
        input_mask = tf.constant(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]], dtype=np.float32)  # input_mask

        outputs, pooled_outputs, all_states, all_attn_weights = model(
            inputs=(input_ids, token_type_ids, input_mask))
        self.assertAllEqual([2, 16, model.hidden_size], outputs.shape)
        self.assertAllEqual([2, model.hidden_size], pooled_outputs.shape)

        self.assertEqual(2, len(all_states))
        for state in all_states:
            self.assertAllEqual([2, 16, model.hidden_size], state.shape)

        self.assertEqual(2, len(all_attn_weights))
        for attention in all_attn_weights:
            self.assertAllEqual([2, model.num_attention_heads, 16, 16], attention.shape)

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

    def testBertModel(self):
        model = BertModel(vocab_size=100, num_layers=2)
        input_ids = tf.constant(
            [1, 2, 3, 4, 5, 6, 7, 5, 3, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5, 6, 6, 6, 7, 7, 8, 0, 0, 0, 0, 0, 0],
            shape=(2, 16),
            dtype=np.int32)  # input_ids
        token_type_ids = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int64)  # token_type_ids,
        input_mask = tf.constant(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]], dtype=np.float32)  # input_mask

        outputs, pooled_outputs = model(inputs=(input_ids, token_type_ids, input_mask))
        self.assertAllEqual([2, 16, 768], outputs.shape)
        self.assertAllEqual([2, 768], pooled_outputs.shape)

    def testBertFroPreTrainingModel(self):
        bert = BertForPretrainingModel(vocab_size=100, num_layers=2)
        input_ids = tf.constant(
            [1, 2, 3, 4, 5, 6, 7, 5, 3, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5, 6, 6, 6, 7, 7, 8, 0, 0, 0, 0, 0, 0],
            shape=(2, 16),
            dtype=np.int32)  # input_ids
        token_type_ids = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int64)  # token_type_ids,
        input_mask = tf.constant(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]], dtype=np.float32)  # input_mask

        predictions, relations = bert(
            inputs=(input_ids, token_type_ids, input_mask))
        self.assertAllEqual([2, 16, 100], predictions.shape)
        self.assertAllEqual([2, 2], relations.shape)

    def testBertModelFromPretrained(self):
        model = BertModel.from_pretrained('/Users/luozhouyang/pretrain-models/bert/chinese_L-12_H-768_A-12')

        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        input_ids = [0, 1, 2, 3, 4, 5, 6, 0, 0, 0]
        input_ids = input_ids + [0] * (512 - len(input_ids))
        input_ids = tf.constant(input_ids, dtype=tf.int32, shape=(1, 512))
        input_mask = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        input_mask = input_mask + [1] * (512 - len(input_mask))
        input_mask = tf.constant(input_mask, dtype=tf.int32, shape=(1, 512))
        segment_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        segment_ids = segment_ids + [0] * (512 - len(segment_ids))
        segment_ids = tf.constant(segment_ids, dtype=tf.int32, shape=(1, 512))

        sequence_outputs, pooled_outputs = model(inputs=(input_ids, input_mask, segment_ids))
        self.assertEqual([1, 512, 768], sequence_outputs.shape)
        self.assertEqual([1, 768], pooled_outputs.shape)
        print(sequence_outputs)
        print(pooled_outputs)

    def testBertFroPretrainingModelFromPretrained(self):
        model = BertForPretrainingModel.from_pretrained(
            '/Users/luozhouyang/pretrain-models/bert/chinese_L-12_H-768_A-12')

        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        input_ids = [0, 1, 2, 3, 4, 5, 6, 0, 0, 0]
        input_ids = input_ids + [0] * (512 - len(input_ids))
        input_ids = tf.constant(input_ids, dtype=tf.int32, shape=(1, 512))
        input_mask = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        input_mask = input_mask + [1] * (512 - len(input_mask))
        input_mask = tf.constant(input_mask, dtype=tf.int32, shape=(1, 512))
        segment_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        segment_ids = segment_ids + [0] * (512 - len(segment_ids))
        segment_ids = tf.constant(segment_ids, dtype=tf.int32, shape=(1, 512))

        predictions, relations = model(inputs=(input_ids, input_mask, segment_ids))
        self.assertEqual([1, 512, 21128], predictions.shape)
        self.assertEqual([1, 2], relations.shape)
        print(predictions)
        print(relations)


if __name__ == "__main__":
    tf.test.main()
