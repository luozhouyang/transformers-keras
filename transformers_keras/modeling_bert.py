import json
import logging
import os

import tensorflow as tf

from transformers_keras.adapters import parse_pretrained_model_files
from transformers_keras.adapters.bert_adapter import BertAdapter

from .modeling_utils import choose_activation


class BertEmbedding(tf.keras.layers.Layer):
    """Embedding layer."""

    def __init__(
        self,
        vocab_size=21128,
        max_positions=512,
        embedding_size=768,
        type_vocab_size=2,
        hidden_dropout_rate=0.1,
        initializer_range=0.02,
        epsilon=1e-12,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert vocab_size > 0, "vocab_size must greater than 0."
        self.vocab_size = vocab_size
        self.max_positions = max_positions
        self.type_vocab_size = type_vocab_size
        self.embedding_size = embedding_size
        self.initializer_range = initializer_range
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=epsilon, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(hidden_dropout_rate)

    def build(self, input_shape):
        self.token_embedding = self.add_weight(
            "word_embeddings",
            shape=[self.vocab_size, self.embedding_size],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range),
        )
        self.position_embedding = self.add_weight(
            "position_embeddings",
            shape=[self.max_positions, self.embedding_size],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range),
        )
        self.token_type_embedding = self.add_weight(
            "token_type_embeddings",
            shape=[self.type_vocab_size, self.embedding_size],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range),
        )
        return super().build(input_shape)

    @property
    def embedding_table(self):
        return self.token_embedding

    def call(self, input_ids, segment_ids=None, position_ids=None, training=None):
        if segment_ids is None:
            segment_ids = tf.zeros_like(input_ids)
        if position_ids is None:
            position_ids = tf.range(0, tf.shape(input_ids)[1], dtype=input_ids.dtype)
            position_ids = tf.expand_dims(position_ids, axis=0)

        position_embeddings = tf.gather(self.position_embedding, position_ids)
        position_embeddings = tf.tile(position_embeddings, multiples=[tf.shape(input_ids)[0], 1, 1])
        token_type_embeddings = tf.gather(self.token_type_embedding, segment_ids)
        token_embeddings = tf.gather(self.token_embedding, input_ids)

        embeddings = token_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings


class BertMultiHeadAtttetion(tf.keras.layers.Layer):
    """Multi head attention."""

    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=8,
        attention_dropout_rate=0.1,
        initializer_range=0.02,
        epsilon=1e-8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.query_weight = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="query",
        )
        self.key_weight = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="key",
        )
        self.value_weight = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="value",
        )
        self.attention_dropout = tf.keras.layers.Dropout(attention_dropout_rate)

    def _split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.hidden_size // self.num_attention_heads))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _scaled_dot_product_attention(self, query, key, value, attention_mask, training=None):
        query = tf.cast(query, dtype=self.dtype)
        key = tf.cast(key, dtype=self.dtype)
        value = tf.cast(value, dtype=self.dtype)

        score = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(query)[-1], self.dtype)
        score = score / tf.math.sqrt(dk)
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, dtype=self.dtype)
            score += tf.cast((1.0 - attention_mask) * -10000.0, dtype=self.dtype)
        attn_weights = tf.nn.softmax(score, axis=-1)
        attn_weights = self.attention_dropout(attn_weights, training=training)
        context = tf.matmul(attn_weights, value)
        return context, attn_weights

    def call(self, query, key, value, attention_mask, training=None):
        batch_size = tf.shape(query)[0]
        query = self._split_heads(self.query_weight(query), batch_size)
        key = self._split_heads(self.key_weight(key), batch_size)
        value = self._split_heads(self.value_weight(value), batch_size)
        context, attn_weights = self._scaled_dot_product_attention(query, key, value, attention_mask, training=training)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, [batch_size, -1, self.hidden_size])
        return context, attn_weights


class BertAttentionOutput(tf.keras.layers.Layer):
    """Attention output layer."""

    def __init__(self, hidden_size=768, hidden_dropout_rate=0.1, initializer_range=0.02, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="dense",
        )
        self.dropout = tf.keras.layers.Dropout(hidden_dropout_rate)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=epsilon, name="LayerNorm")

    def call(self, input_states, hidden_states, training=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.layernorm(hidden_states + input_states)
        return hidden_states


class BertAttention(tf.keras.layers.Layer):
    """Bert attention."""

    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=8,
        hidden_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        initializer_range=0.02,
        epsilon=1e-5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.attention = BertMultiHeadAtttetion(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout_rate=attention_dropout_rate,
            initializer_range=initializer_range,
            epsilon=epsilon,
            name="self",
        )
        self.attention_output = BertAttentionOutput(
            hidden_size=hidden_size,
            hidden_dropout_rate=hidden_dropout_rate,
            initializer_range=initializer_range,
            epsilon=epsilon,
            name="output",
        )

    def call(self, hidden_states, attention_mask, training=None):
        context, attention_weights = self.attention(
            hidden_states, hidden_states, hidden_states, attention_mask, training=training
        )
        outputs = self.attention_output(hidden_states, context, training=training)
        return outputs, attention_weights


class BertIntermediate(tf.keras.layers.Layer):
    """Bert intermediate."""

    def __init__(self, intermediate_size=3072, activation="gelu", initializer_range=0.02, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            intermediate_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="dense",
        )
        self.activation = choose_activation(activation)

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class BertIntermediateOutput(tf.keras.layers.Layer):
    """Bert intermediate output."""

    def __init__(self, hidden_size=768, hidden_dropout_rate=0.1, initializer_range=0.02, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="dense",
        )
        self.dropout = tf.keras.layers.Dropout(hidden_dropout_rate)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=epsilon, name="LayerNorm")

    def call(self, input_states, hidden_states, training=None):
        hidden_states = self.dropout(self.dense(hidden_states), training=training)
        hidden_states = self.layernorm(hidden_states + input_states)
        return hidden_states


class BertEncoderLayer(tf.keras.layers.Layer):
    """Encoder layer."""

    def __init__(
        self,
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
        super().__init__(**kwargs)
        # attention block
        self.attention = BertAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            hidden_dropout_rate=hidden_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            initializer_range=initializer_range,
            name="attention",
        )
        # intermediate block
        self.intermediate = BertIntermediate(
            intermediate_size=intermediate_size,
            activation=activation,
            initializer_range=initializer_range,
            name="intermediate",
        )
        # output block
        self.intermediate_output = BertIntermediateOutput(
            hidden_size=hidden_size,
            hidden_dropout_rate=hidden_dropout_rate,
            initializer_range=initializer_range,
            epsilon=epsilon,
            name="output",
        )

    def call(self, hidden_states, attn_mask, training=None):
        attn_output, attn_weights = self.attention(hidden_states, attn_mask, training=training)
        outputs = self.intermediate(attn_output)
        outputs = self.intermediate_output(attn_output, outputs, training=training)
        return outputs, attn_weights


class BertEncoder(tf.keras.layers.Layer):
    """Encoder."""

    def __init__(
        self,
        num_layers=6,
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
        super().__init__(**kwargs)
        self.encoder_layers = [
            BertEncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                activation=activation,
                hidden_dropout_rate=hidden_dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                epsilon=epsilon,
                initializer_range=initializer_range,
                name="layer_{}".format(i),
            )
            for i in range(num_layers)
        ]

    def call(self, hidden_states, attention_mask, training=None):
        all_hidden_states = []
        all_attention_scores = []
        for _, encoder in enumerate(self.encoder_layers):
            hidden_states, attention_score = encoder(hidden_states, attention_mask, training=training)
            all_hidden_states.append(hidden_states)
            all_attention_scores.append(attention_score)
        # stack all_hidden_states to shape:
        # [batch_size, num_layers, num_attention_heads, hidden_size]
        all_hidden_states = tf.stack(all_hidden_states, axis=0)
        all_hidden_states = tf.transpose(all_hidden_states, perm=[1, 0, 2, 3])
        # stack all_attention_scores to shape:
        # [batch_size, num_layers, num_attention_heads, seqeucen_length, sequence_length]
        all_attention_scores = tf.stack(all_attention_scores, axis=0)
        all_attention_scores = tf.transpose(all_attention_scores, perm=[1, 0, 2, 3, 4])
        return hidden_states, all_hidden_states, all_attention_scores


class BertPooler(tf.keras.layers.Layer):
    """Pooler."""

    def __init__(self, hidden_size=768, initializer_range=0.02, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=initializer_range),
            activation="tanh",
            name="dense",
        )

    def call(self, inputs):
        hidden_states = inputs
        # pool the first token: [CLS]
        outputs = self.dense(hidden_states[:, 0])
        return outputs


class BertModel(tf.keras.layers.Layer):
    """Bert main model."""

    def __init__(
        self,
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
        super().__init__(**kwargs)
        self.bert_embedding = BertEmbedding(
            vocab_size=vocab_size,
            max_positions=max_positions,
            embedding_size=hidden_size,
            type_vocab_size=type_vocab_size,
            hidden_dropout_rate=hidden_dropout_rate,
            initializer_range=initializer_range,
            epsilon=epsilon,
            name="embeddings",
        )

        self.bert_encoder = BertEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            activation=activation,
            hidden_dropout_rate=hidden_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            epsilon=epsilon,
            initializer_range=initializer_range,
            name="encoder",
        )

        self.bert_pooler = BertPooler(hidden_size=hidden_size, initializer_range=initializer_range, name="pooler")

    def get_embedding_table(self):
        return self.bert_embedding.embedding_table

    def call(self, input_ids, segment_ids, attention_mask, training=None):
        embedding = self.bert_embedding(input_ids, segment_ids, training=training)
        # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
        attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
        output, all_hidden_states, all_attention_scores = self.bert_encoder(
            embedding, attention_mask, training=training
        )
        pooled_output = self.bert_pooler(output)
        return output, pooled_output, all_hidden_states, all_attention_scores


class BertPretrainedModel(tf.keras.Model):
    """Base class for all pretrained model. Can not used to initialize an instance directly."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_dir,
        override_params=None,
        use_functional_api=True,
        adapter=None,
        check_weights=True,
        verbose=True,
        **kwargs
    ):
        """Load pretrained bert weights.

        Args:
            pretrained_model_dir: A file path to pretrained model dir, including ckpt and config files.
            override_params: Add extra model params or override pretrained model's params to constructor
            use_functional_api: Python boolean, default is True.
                When subclassing BertPretrainedModel, you may construct model using Keras' functional API,
                then you should set use_functional_api=True, otherwise, you should set use_functional_api=False.
            adapter: An adpater to adapte pretrained weights to this new created model. Default is `BertAdapter`.
                In most case, you do not need to specify a `adapter`.
            check_weights: Python boolean. If true, check model weights' value after loading weights from ckpt
            verbose: Python boolean.If True, logging more detailed informations when loadding pretrained weights
        """

        config_file, ckpt, _ = parse_pretrained_model_files(pretrained_model_dir)
        if not adapter:
            adapter = BertAdapter(
                skip_token_embedding=kwargs.pop("skip_token_embedding", False),
                skip_position_embedding=kwargs.pop("skip_position_embedding", False),
                skip_segment_embedding=kwargs.pop("skip_segment_embedding", False),
                skip_embedding_layernorm=kwargs.pop("skip_embedding_layernorm", False),
                skip_pooler=kwargs.pop("skip_pooler", False),
            )
        model_config = adapter.adapte_config(config_file, **kwargs)
        if override_params:
            model_config.update(override_params)
        logging.info("Load model config: \n%s", json.dumps(model_config, indent=4))
        model = cls(**model_config, **kwargs)
        assert (
            getattr(model, "bert_model", None) is not None
        ), "BertPretrainedModel must have an attribute named bert_model!"
        inputs = model.dummy_inputs()
        model(inputs=list(inputs), training=False)
        adapter.adapte_weights(
            bert=model.bert_model,
            config=model_config,
            ckpt=ckpt,
            prefix="" if use_functional_api else model.name,
            check_weights=check_weights,
            verbose=verbose,
            **kwargs
        )
        return model

    @classmethod
    def from_config_file(cls, config_file, override_params=None, adapter=None, **kwargs):
        if not adapter:
            adapter = BertAdapter(
                skip_token_embedding=kwargs.pop("skip_token_embedding", False),
                skip_position_embedding=kwargs.pop("skip_position_embedding", False),
                skip_segment_embedding=kwargs.pop("skip_segment_embedding", False),
                skip_embedding_layernorm=kwargs.pop("skip_embedding_layernorm", False),
                skip_pooler=kwargs.pop("skip_pooler", False),
            )
        model_config = adapter.adapte_config(config_file, **kwargs)
        if override_params:
            model_config.update(override_params)
        logging.info("Load model config: \n%s", json.dumps(model_config, indent=4))
        model = cls(**model_config, **kwargs)
        assert (
            getattr(model, "bert_model", None) is not None
        ), "BertPretrainedModel must have an attribute named bert_model!"
        inputs = model.dummy_inputs()
        model(inputs=list(inputs), training=False)
        return model

    def dummy_inputs(self):
        input_ids = tf.constant([0] * 128, dtype=tf.int32, shape=(1, 128))
        segment_ids = tf.constant([0] * 128, dtype=tf.int32, shape=(1, 128))
        attn_mask = tf.constant([1] * 128, dtype=tf.int32, shape=(1, 128))
        return input_ids, segment_ids, attn_mask


class Bert(BertPretrainedModel):
    """Bert, can load pretrained weights."""

    def __init__(
        self,
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
        return_states=False,
        return_attention_weights=False,
        **kwargs
    ):
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
        sequence_output, pooled_output, hidden_states, attention_weights = bert_model(
            input_ids, segment_ids, attention_mask
        )
        outputs = [
            tf.keras.layers.Lambda(lambda x: x, name="sequence_output")(sequence_output),
            tf.keras.layers.Lambda(lambda x: x, name="pooled_output")(pooled_output),
        ]
        if return_states:
            outputs += [tf.keras.layers.Lambda(lambda x: x, name="hidden_states")(hidden_states)]
        if return_attention_weights:
            outputs += [tf.keras.layers.Lambda(lambda x: x, name="attention_weights")(attention_weights)]

        super().__init__(inputs=[input_ids, segment_ids, attention_mask], outputs=outputs, **kwargs)

        self.bert_model = bert_model
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.max_positions = max_positions
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.initializer_range = initializer_range
        self.initialize_range = initializer_range
        self.return_states = return_states
        self.return_attention_weights = return_attention_weights

    def get_config(self):
        config = {
            "vocab_size": self.vocab_size,
            "type_vocab_size": self.type_vocab_size,
            "max_positions": self.max_positions,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_dropout_rate": self.hidden_dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "initializer_range": self.initializer_range,
            "return_states": self.return_states,
            "return_attention_weights": self.return_attention_weights,
        }
        return config
