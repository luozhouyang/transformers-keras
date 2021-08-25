import tensorflow as tf
import tensorflow_addons as tfa
from transformers_keras.modeling_albert import (AlbertModel,
                                                AlbertPretrainedModel)
from transformers_keras.modeling_bert import BertModel, BertPretrainedModel


def _unpack_data(data):
    """Support sample_weight"""
    if len(data) == 3:
        x, y, sample_weight = data
    elif len(data) == 2:
        x, y, sample_weight = data[0], data[1], None
    elif len(data) == 1:
        x, y, sample_weight = data[0], None, None
    return x, y, sample_weight


class CRFModel(tf.keras.Model):
    """Wrappe a CRF layer for base model."""

    def __init__(
        self,
        model: tf.keras.Model,
        units: int,
        chain_initializer="orthogonal",
        use_boundary: bool = True,
        boundary_initializer="zeros",
        use_kernel: bool = True,
        **kwargs
    ):
        # build functional model
        crf = tfa.layers.CRF(
            units=units,
            chain_initializer=chain_initializer,
            use_boundary=use_boundary,
            boundary_initializer=boundary_initializer,
            use_kernel=use_kernel,
            **kwargs
        )
        # take model's first output passed to CRF layer
        decode_sequence, potentials, sequence_length, kernel = crf(inputs=model.outputs[0])
        # set name for outputs
        decode_sequence = tf.keras.layers.Lambda(lambda x: x, name="decode_sequence")(decode_sequence)
        potentials = tf.keras.layers.Lambda(lambda x: x, name="potentials")(potentials)
        sequence_length = tf.keras.layers.Lambda(lambda x: x, name="sequence_length")(sequence_length)
        kernel = tf.keras.layers.Lambda(lambda x: x, name="kernel")(kernel)
        super().__init__(inputs=model.inputs, outputs=[decode_sequence, potentials, sequence_length, kernel], **kwargs)
        self.crf = crf

    def train_step(self, data):
        x, y, sample_weight = _unpack_data(data)
        with tf.GradientTape() as tape:
            decode_sequence, potentials, sequence_length, kernel = self(x, training=True)
            crf_loss = -tfa.text.crf_log_likelihood(potentials, y, sequence_length, kernel)[0]
            if sample_weight is not None:
                crf_loss = crf_loss * sample_weight
            crf_loss = tf.reduce_mean(crf_loss)
            loss = crf_loss + sum(self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, potentials)
        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

    def test_step(self, data):
        x, y, sample_weight = _unpack_data(data)
        decode_sequence, potentials, sequence_length, kernel = self(x, training=False)
        crf_loss = -tfa.text.crf_log_likelihood(potentials, y, sequence_length, kernel)[0]
        if sample_weight is not None:
            crf_loss = crf_loss * sample_weight
        crf_loss = tf.reduce_mean(crf_loss)
        loss = crf_loss + sum(self.losses)
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, potentials)
        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

    def predict_step(self, data):
        x, _, _ = _unpack_data(data)
        decode_sequence, potentials, sequence_length, kernel = self(x, training=False)
        return decode_sequence


class BertCRFForTokenClassification(BertPretrainedModel):
    """BERT+CRF use for token classification."""

    def __init__(
        self,
        num_labels,
        vocab_size=21128,
        max_positions=512,
        hidden_size=768,
        type_vocab_size=2,
        num_layers=6,
        num_attention_heads=8,
        intermediate_size=3072,
        activation="gelu",
        hidden_dropout_rate=0.2,
        attention_dropout_rate=0.1,
        initializer_range=0.02,
        epsilon=1e-12,
        **kwargs
    ):
        # build functional model
        input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
        segment_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="segment_ids")
        attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
        bert_model = BertModel(
            vocab_size=vocab_size,
            max_positions=max_positions,
            hidden_size=hidden_size,
            type_vocab_size=type_vocab_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            activation=activation,
            hidden_dropout_rate=hidden_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            initializer_range=initializer_range,
            epsilon=epsilon,
            name="bert",
        )
        crf = tfa.layers.CRF(num_labels)
        sequence_outputs, _, _, _ = bert_model(input_ids, segment_ids, attention_mask)
        decode_sequence, potentials, sequence_length, kernel = crf(sequence_outputs)
        super().__init__(
            inputs=[input_ids, segment_ids, attention_mask],
            outputs=[decode_sequence, potentials, sequence_length, kernel],
            **kwargs
        )

        self.num_labels = num_labels
        self.bert_model = bert_model
        self.crf = crf

    @tf.function( # fmt: skip
        input_signature=[{
            "input_ids": tf.TensorSpec(shape=(None, None,), dtype=tf.int32, name="input_ids"), # fmt: skip
            "segment_ids": tf.TensorSpec(shape=(None, None,), dtype=tf.int32, name="segment_ids"), # fmt: skip
            "attention_mask": tf.TensorSpec(shape=(None, None,), dtype=tf.int32, name="attention_mask"), # fmt: skip
        }]
    )
    def forward(self, inputs):
        input_ids, segment_ids, attention_mask = inputs["input_ids"], inputs["segment_ids"], inputs["attention_mask"]
        sequence, _, _, _ = self(inputs=[input_ids, segment_ids, attention_mask], training=False)
        return {"decode_sequence": sequence}

    def train_step(self, data):
        x, y, sample_weight = _unpack_data(data)
        input_ids, segment_ids, attention_mask = x["input_ids"], x["segment_ids"], x["attention_mask"]
        with tf.GradientTape() as tape:
            decode_sequence, potentials, sequence_length, kernel = self(
                inputs=[input_ids, segment_ids, attention_mask], training=True
            )
            crf_loss = -tfa.text.crf_log_likelihood(potentials, y, sequence_length, kernel)[0]
            if sample_weight is not None:
                crf_loss = crf_loss * sample_weight
            crf_loss = tf.reduce_mean(crf_loss)
            loss = crf_loss + sum(self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, potentials)
        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

    def test_step(self, data):
        x, y, sample_weight = _unpack_data(data)
        input_ids, segment_ids, attention_mask = x["input_ids"], x["segment_ids"], x["attention_mask"]
        decode_sequence, potentials, sequence_length, kernel = self(
            inputs=[input_ids, segment_ids, attention_mask], training=False
        )
        crf_loss = -tfa.text.crf_log_likelihood(potentials, y, sequence_length, kernel)[0]
        if sample_weight is not None:
            crf_loss = crf_loss * sample_weight
        crf_loss = tf.reduce_mean(crf_loss)
        loss = crf_loss + sum(self.losses)
        self.compiled_metrics.update_state(y, potentials)
        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

    def predict_step(self, data):
        x, _, _ = _unpack_data(data)
        input_ids, segment_ids, attention_mask = x["input_ids"], x["segment_ids"], x["attention_mask"]
        decode_sequence, potentials, sequence_length, kernel = self(
            inputs=[input_ids, segment_ids, attention_mask], training=False)
        return decode_sequence



class AlertCRFForTokenClassification(AlbertPretrainedModel):
    """ALBERT+CRF use for token classification."""

    def __init__(
        self,
        num_labels,
        vocab_size=21128,
        max_positions=512,
        embedding_size=128,
        type_vocab_size=2,
        num_layers=12,
        num_groups=1,
        num_layers_each_group=1,
        hidden_size=768,
        num_attention_heads=8,
        intermediate_size=3072,
        activation="gelu",
        hidden_dropout_rate=0.2,
        attention_dropout_rate=0.1,
        epsilon=1e-12,
        initializer_range=0.02,
        **kwargs
    ):
        # build functional model
        input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
        segment_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="segment_ids")
        attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
        albert_model = AlbertModel(
            vocab_size=vocab_size,
            max_positions=max_positions,
            embedding_size=embedding_size,
            type_vocab_size=type_vocab_size,
            num_layers=num_layers,
            num_groups=num_groups,
            num_layers_each_group=num_layers_each_group,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            activation=activation,
            hidden_dropout_rate=hidden_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            epsilon=epsilon,
            initializer_range=initializer_range,
            name="albert",
        )
        crf = tfa.layers.CRF(num_labels)
        sequence_outputs, _, _, _ = albert_model(input_ids, segment_ids, attention_mask)
        decode_sequence, potentials, sequence_length, kernel = crf(sequence_outputs)
        super().__init__(
            inputs=[input_ids, segment_ids, attention_mask],
            outputs=[decode_sequence, potentials, sequence_length, kernel],
            **kwargs
        )

        self.num_labels = num_labels
        self.albert_model = albert_model
        self.crf = crf

    @tf.function(
        input_signature=[{
            "input_ids": tf.TensorSpec(shape=(None, None,), dtype=tf.int32, name="input_ids"), # fmt: skip
            "segment_ids": tf.TensorSpec(shape=(None, None,), dtype=tf.int32, name="segment_ids"), # fmt: skip
            "attention_mask": tf.TensorSpec(shape=(None, None,), dtype=tf.int32, name="attention_mask"), # fmt: skip
        }]
    )
    def forward(self, inputs):
        input_ids, segment_ids, attention_mask = inputs["input_ids"], inputs["segment_ids"], inputs["attention_mask"]
        sequence, _, _, _ = self(inputs=[input_ids, segment_ids, attention_mask], training=False)
        return {"decode_sequence": sequence}

    def train_step(self, data):
        x, y, sample_weight = _unpack_data(data)
        input_ids, segment_ids, attention_mask = x["input_ids"], x["segment_ids"], x["attention_mask"]
        with tf.GradientTape() as tape:
            decode_sequence, potentials, sequence_length, kernel = self(
                inputs=[input_ids, segment_ids, attention_mask], training=True
            )
            crf_loss = -tfa.text.crf_log_likelihood(potentials, y, sequence_length, kernel)[0]
            if sample_weight is not None:
                crf_loss = crf_loss * sample_weight
            crf_loss = tf.reduce_mean(crf_loss)
            loss = crf_loss + sum(self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, potentials)
        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

    def test_step(self, data):
        x, y, sample_weight = _unpack_data(data)
        input_ids, segment_ids, attention_mask = x["input_ids"], x["segment_ids"], x["attention_mask"]
        decode_sequence, potentials, sequence_length, kernel = self(
            inputs=[input_ids, segment_ids, attention_mask], training=False
        )
        crf_loss = -tfa.text.crf_log_likelihood(potentials, y, sequence_length, kernel)[0]
        if sample_weight is not None:
            crf_loss = crf_loss * sample_weight
        crf_loss = tf.reduce_mean(crf_loss)
        loss = crf_loss + sum(self.losses)
        self.compiled_metrics.update_state(y, potentials)
        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results
        
    def predict_step(self, data):
        x, _, _ = _unpack_data(data)
        input_ids, segment_ids, attention_mask = x["input_ids"], x["segment_ids"], x["attention_mask"]
        decode_sequence, potentials, sequence_length, kernel = self(
            inputs=[input_ids, segment_ids, attention_mask], training=False)
        return decode_sequence

