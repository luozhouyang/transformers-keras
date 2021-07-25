import tensorflow as tf
from transformers_keras.modeling_bert import BertModel, BertPretrainedModel


class BertForSequenceClassification(BertPretrainedModel):

    def __init__(self,
                 num_labels=2,
                 vocab_size=21128,
                 max_positions=512,
                 hidden_size=768,
                 type_vocab_size=2,
                 num_layers=6,
                 num_attention_heads=8,
                 intermediate_size=3072,
                 activation='gelu',
                 hidden_dropout_rate=0.2,
                 attention_dropout_rate=0.1,
                 initializer_range=0.02,
                 epsilon=1e-12,
                 **kwargs):
        input_ids = tf.keras.layers.Input(shape=(None, ), dtype=tf.int32, name='input_ids')
        segment_ids = tf.keras.layers.Input(shape=(None, ), dtype=tf.int32, name='segment_ids')
        attention_mask = tf.keras.layers.Input(shape=(None, ), dtype=tf.int32, name='attention_mask')
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
            name='bert')

        _, pooled_output, _, _ = bert_model(input_ids, segment_ids, attention_mask)
        logits = tf.keras.layers.Dense(num_labels, name='dense')(pooled_output)
        super().__init__(
            inputs=[input_ids, segment_ids, attention_mask],
            outputs=[logits],
            **kwargs)

        self.bert_model = bert_model
        self.num_labels = num_labels
