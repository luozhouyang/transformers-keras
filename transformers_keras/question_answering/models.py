import tensorflow as tf
from transformers_keras.modeling_albert import AlbertModel, AlbertPretrainedModel
from transformers_keras.modeling_bert import BertModel, BertPretrainedModel


class BertForQuestionAnswering(BertPretrainedModel):
    """Bert for question answering"""

    def __init__(
        self,
        num_labels=2,
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

        sequence_output, _, _, _ = bert_model(input_ids, segment_ids, attention_mask)
        logits = tf.keras.layers.Dense(num_labels, name="dense")(sequence_output)
        head_logits = tf.keras.layers.Lambda(lambda x: x[:, :, 0], name="head")(logits)
        tail_logits = tf.keras.layers.Lambda(lambda x: x[:, :, 1], name="tail")(logits)

        super().__init__(inputs=[input_ids, segment_ids, attention_mask], outputs=[head_logits, tail_logits], **kwargs)

        self.num_labels = num_labels
        self.bert_model = bert_model


class AlbertForQuestionAnswering(AlbertPretrainedModel):
    """Albert for question answering"""

    def __init__(
        self,
        num_labels=2,
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

        sequence_output, _, _, _ = albert_model(input_ids, segment_ids, attention_mask)
        logits = tf.keras.layers.Dense(num_labels, name="dense")(sequence_output)
        head_logits = tf.keras.layers.Lambda(lambda x: x[:, :, 0], name="head")(logits)
        tail_logits = tf.keras.layers.Lambda(lambda x: x[:, :, 1], name="tail")(logits)

        super().__init__(inputs=[input_ids, segment_ids, attention_mask], outputs=[head_logits, tail_logits], **kwargs)

        self.num_labels = num_labels
        self.albert_model = albert_model


class BertForQuestionAnsweringX(BertPretrainedModel):
    """Bert for question asnwering, with classification head."""

    def __init__(
        self,
        num_labels=2,
        num_classes=2,
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

        sequence_output, pooled_output, _, _ = bert_model(input_ids, segment_ids, attention_mask)
        logits = tf.keras.layers.Dense(num_labels, name="dense")(sequence_output)
        head_logits = tf.keras.layers.Lambda(lambda x: x[:, :, 0], name="head")(logits)
        tail_logits = tf.keras.layers.Lambda(lambda x: x[:, :, 1], name="tail")(logits)
        class_logits = tf.keras.layers.Dense(num_classes, name="class")(pooled_output)

        super().__init__(
            inputs=[input_ids, segment_ids, attention_mask], outputs=[head_logits, tail_logits, class_logits], **kwargs
        )

        self.num_labels = num_labels
        self.num_classes = num_classes
        self.bert_model = bert_model


class AlbertForQuestionAnsweringX(BertPretrainedModel):
    """Albert for question asnwering, with classification head."""

    def __init__(
        self,
        num_labels=2,
        num_classes=2,
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

        sequence_output, pooled_output, _, _ = albert_model(input_ids, segment_ids, attention_mask)
        logits = tf.keras.layers.Dense(num_labels, name="dense")(sequence_output)
        head_logits = tf.keras.layers.Lambda(lambda x: x[:, :, 0], name="head")(logits)
        tail_logits = tf.keras.layers.Lambda(lambda x: x[:, :, 1], name="tail")(logits)
        class_logits = tf.keras.layers.Dense(num_classes, name="class")(pooled_output)

        super().__init__(
            inputs=[input_ids, segment_ids, attention_mask], outputs=[head_logits, tail_logits, class_logits], **kwargs
        )

        self.num_labels = num_labels
        self.num_classes = num_classes
        self.albert_model = albert_model
