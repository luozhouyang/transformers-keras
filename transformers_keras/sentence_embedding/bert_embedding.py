import tensorflow as tf
from transformers_keras.modeling_bert import BertModel, BertPretrainedModel


class BertForSentenceEmbedding(BertPretrainedModel):
    """Bert for sentence embedding."""

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
        sequence_output, pooled_output, hidden_states, attention_weights = bert_model(
            input_ids, segment_ids, attention_mask
        )

        cls_embedding = tf.keras.layers.Lambda(lambda x: x[:, 0, :], name="cls")(sequence_output)
        pooler_embedding = tf.keras.layers.Lambda(lambda x: x, name="pooler")(pooled_output)
        avg_embedding = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1), name="avg")(sequence_output)
        all_hidden_states = tf.keras.layers.Lambda(lambda x: x, name="hidden_states")(hidden_states)
        all_attention_weights = tf.keras.layers.Lambda(lambda x: x, name="attention_weights")(attention_weights)
        super().__init__(
            inputs=[input_ids, segment_ids, attention_mask],
            outputs=[cls_embedding, pooler_embedding, avg_embedding, all_hidden_states, all_attention_weights],
            **kwargs
        )

        self.bert_model = bert_model
