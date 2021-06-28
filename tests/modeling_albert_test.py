import numpy as np
import tensorflow as tf
from transformers_keras.modeling_albert import (Albert, AlbertEmbedding,
                                                AlbertEncoder,
                                                AlbertEncoderGroup,
                                                AlbertEncoderLayer)


class ModelingAlbertTest(tf.test.TestCase):

    def testAlbertEmbedding(self):
        embedding = AlbertEmbedding(vocab_size=100, embedding_size=128)
        inputs = (
            tf.constant([[0, 2, 3, 4, 5, 1]]),  # input_ids
            tf.constant([[0, 0, 0, 1, 1, 1]]),  # token_type_ids
            # tf.constant([[1, 1, 1, 1, 1, 1]]),  # attention_mask
        )
        outputs = embedding(inputs[0], inputs[1], training=True)
        self.assertAllEqual([1, 6, 128], outputs.shape)

    def testAlbertEncoderLayer(self):
        encoder = AlbertEncoderLayer()
        hidden_states = tf.random.uniform(shape=(2, 16, 768))  # hidden states
        mask = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]],
                           shape=(2, 16), dtype=tf.float32)  # mask
        attention_mask = mask[:, tf.newaxis, tf.newaxis, :]
        outputs, attn_weights = encoder(hidden_states, attention_mask)
        self.assertAllEqual([2, 16, 768], outputs.shape)
        self.assertAllEqual([2, 8, 16, 16], attn_weights.shape)

    def testAlbertEncoderGroup(self):

        hidden_states = tf.random.uniform(shape=(2, 16, 768))  # hidden states
        mask = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]],
                           shape=(2, 16), dtype=tf.float32)  # mask
        mask = mask[:, tf.newaxis, tf.newaxis, :]

        def _run_albert_encoder_group(num_layers_each_group):
            group = AlbertEncoderGroup(num_layers_each_group=num_layers_each_group)
            outputs, group_states, group_attn_weights = group(hidden_states, mask)
            self.assertAllEqual([2, 16, 768], outputs.shape)

            self.assertEqual(num_layers_each_group, len(group_states))
            for state in group_states:
                self.assertAllEqual([2, 16, 768], state.shape)

            self.assertEqual(num_layers_each_group, len(group_attn_weights))
            for attention in group_attn_weights:
                self.assertAllEqual([2, 8, 16, 16], attention.shape)

        for layer in [1, 2, 3, 4]:
            _run_albert_encoder_group(layer)

    def testAlbertEncoder(self):
        hidden_states = tf.random.uniform(shape=(2, 16, 768))  # hidden states
        mask = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]],
                           shape=(2, 16), dtype=tf.float32)  # mask
        mask = mask[:, tf.newaxis, tf.newaxis, :]

        NUM_LAYERS = 4
        NUM_GROUPS = 1
        encoder = AlbertEncoder(num_layers=NUM_LAYERS, num_groups=NUM_GROUPS, num_layers_each_group=1, hidden_size=768)

        outputs, all_states, all_attn_weights = encoder(hidden_states, mask)
        self.assertAllEqual([2, 16, 768], outputs.shape)

        self.assertAllEqual([2, NUM_LAYERS * NUM_GROUPS, 16, 768], all_states.shape)
        self.assertAllEqual([2, NUM_LAYERS * NUM_GROUPS, 8, 16, 16], all_attn_weights.shape)

    def _build_albert_inputs(self):
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

    def _check_albert_outputs(self, return_states=False, return_attention_weights=False):
        NUM_LAYERS, NUM_GROUPS, NUM_LAYERS_EACH_GROUP = 4, 1, 1
        model = Albert(
            vocab_size=100, hidden_size=768,
            num_layers=NUM_LAYERS, num_groups=NUM_GROUPS, num_layers_each_group=NUM_LAYERS_EACH_GROUP,
            return_states=return_states,
            return_attention_weights=return_attention_weights)
        input_ids, segment_ids, attn_mask = self._build_albert_inputs()
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
            self.assertAllEqual([2, NUM_LAYERS, 16, 768], all_states.shape)

        if all_attn_weights is not None:
            self.assertAllEqual([2, NUM_LAYERS, 8, 16, 16], all_attn_weights.shape)

    def testAlbert(self):
        self._check_albert_outputs(return_states=True, return_attention_weights=True)
        self._check_albert_outputs(return_states=True, return_attention_weights=False)
        self._check_albert_outputs(return_states=False, return_attention_weights=True)
        self._check_albert_outputs(return_states=False, return_attention_weights=False)

    def test_build_model(self):
        model = Albert(vocab_size=21128)
        input_ids, segment_ids, attn_mask = model.dummy_inputs()
        model(input_ids, segment_ids, attn_mask)
        model.summary()

        for v in model.trainable_weights:
            print(v.name)

    def test_albert_config(self):
        model = Albert(vocab_size=21128)
        print(model.get_config())

    def test_export_saved_model(self):
        model = Albert(vocab_size=21128, num_layers=12, num_attention_heads=8,
                       return_states=True, return_attention_weights=True)
        input_ids, segment_ids, input_mask = model.dummy_inputs()
        model(input_ids=input_ids, segment_ids=segment_ids, attention_mask=input_mask)
        model.summary()
        model.save('models/albert/export/1', include_optimizer=False, signatures=model.serving)

    def test_load_saved_model(self):
        loaded = tf.saved_model.load('models/albert/export/1')
        model = loaded.signatures['serving_default']
        print(model.structured_input_signature)
        print(model.structured_outputs)


if __name__ == "__main__":
    tf.test.main()
