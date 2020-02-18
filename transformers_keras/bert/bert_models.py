import tensorflow as tf

from transformers_keras.bert.bert_embedding import BertEmbedding
from transformers_keras.bert.bert_encoder import ACT2FN, BertEncoder


class BertPooler(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            activation='tanh',
            name='pooler'
        )

    def call(self, inputs, training=False):
        hidden_states = inputs
        # pool the first token: [CLS]
        outputs = self.dense(hidden_states[:, 0])
        return outputs


class BertModel(tf.keras.Model):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.bert_embedding = BertEmbedding(config, **kwargs)
        self.bert_encoder = BertEncoder(config, **kwargs)
        self.bert_pooler = BertPooler(config, **kwargs)

    def call(self, inputs, training=False):
        input_ids, position_ids, token_type_ids, attention_mask = inputs
        embedding = self.bert_embedding(inputs=[input_ids, position_ids, token_type_ids], training=training)
        all_hidden_states, all_attention_scores = self.bert_encoder(
            inputs=[embedding, attention_mask], training=training)
        last_hidden_state = all_hidden_states[-1]
        output = self.bert_pooler(last_hidden_state)

        return {
            'last_hidden_state': last_hidden_state,
            'pooled_output': output,
            'all_hidden_states': all_hidden_states,
            'all_attention_scores': all_attention_scores,
        }


class BertMLMHead(tf.keras.layers.Layer):
    """Masked language model for BERT pre-training."""

    def __init__(self, config, embedding, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.embedding = embedding
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name='dense'
        )
        if isinstance(config.hidden_activation, tf.keras.layers.Activation):
            self.activation = config.hidden_activation
        elif isinstance(config.hidden_activation, str):
            self.activation = ACT2FN[config.hidden_activation]
        else:
            self.activation = ACT2FN['gelu']

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12)

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,), initializer='zeros', trainable=True, name='bias')
        super().build(input_shape)

    def call(self, inputs, training=False):
        hidden_states = inputs
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.embedding(inputs=hidden_states, mode='linear')
        hidden_states = hidden_states + self.bias
        return hidden_states


class BertNSPHead(tf.keras.layers.Layer):
    """Next sentence prediction for BERT pre-training."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.classifier = tf.keras.layers.Dense(
            2,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name='sequence_relationip'
        )

    def call(self, inputs, training=False):
        pooled_output = inputs
        relation = self.classifier(pooled_output)
        return relation


class Bert4PreTrainingModel(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.bert = BertModel(config, name='Bert')
        self.mlm = BertMLMHead(config, self.bert.bert_embedding, name='MLM')
        self.nsp = BertNSPHead(config, name='NSP')

    def call(self, inputs, training=False):
        outputs = self.bert(inputs, training=training)
        sequence_output = outputs['last_hidden_state']
        pooled_output = outputs['pooled_output']
        prediction_scores = self.mlm(sequence_output, training=training)
        relation_scores = self.nsp(pooled_output)
        return {
            'prediction_scores': prediction_scores,
            'relation_scores': relation_scores,
            'last_hidden_state': sequence_output,
            'all_hidden_states': outputs['all_hidden_states'],
            'all_attention_scores': outputs['all_attention_scores']
        }


class Bert4MaskedLM(tf.keras.Model):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.bert = BertModel(config, name='Bert')
        self.mlm = BertMLMHead(config, self.bert.bert_embedding, name='MLM')

    def call(self, inputs, training=False):
        outputs = self.bert(inputs, traning=training)
        sequence_output = outputs['last_hidden_state']
        prediction_scores = self.mlm(sequence_output, training=training)
        return {
            'prediction_scores': prediction_scores,
            'all_hidden_states': outputs['all_hidden_state'],
            'all_attention_scores': outputs['all_attention_scores']
        }


class Bert4NextSetencePredication(tf.keras.Model):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.bert = BertModel(config, name='Bert')
        self.nsp = BertNSPHead(config, name='NSP')

    def call(self, inputs, training=False):
        outputs = self.bert(inputs, training=training)
        pooled_output = outputs['pooled_output']
        relation_scores = self.nsp(pooled_output)
        return {
            'relation_scores': relation_scores,
            'all_hidden_states': outputs['all_hidden_states'],
            'all_attention_scores': outputs['all_attention_scores']
        }
