import tensorflow as tf
from transformers_keras.modeling_bert import BertModel, BertPretrainedModel


class AbstractSimCSE(BertPretrainedModel):

    def __init__(self,
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
        # build functional model
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
        sequence_output, _, _, _ = bert_model(input_ids, segment_ids, attention_mask)
        embedding = tf.keras.layers.Dense(hidden_size, name='embedding')(sequence_output[:, 0, :])
        super().__init__(inputs=[input_ids, segment_ids, attention_mask], outputs=[embedding])
        self.bert_model = bert_model

    def forward(self, x, y, training=False):
        raise NotImplementedError()

    def train_step(self, data):
        x = data
        batch_size = tf.shape(x['input_ids'])[0]
        y_true = tf.range(0, batch_size)
        with tf.GradientTape() as tape:
            y_pred, loss = self.forward(x, y_true, training=True)

        # Compute gradients
        trainable_vars = self.trainable_variables
        # print_variables(trainable_vars)
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y_true, y_pred)
        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': loss})
        return results

    def test_step(self, data):
        x = data
        batch_size = tf.shape(x['input_ids'])[0]
        y_true = tf.range(0, batch_size)
        y_pred, loss = self.forward(x, y_true, training=False)
        self.compiled_metrics.update_state(y_true, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': loss})
        return results


class UnsupervisedSimCSE(AbstractSimCSE):

    def forward(self, x, y, training=False):
        input_ids, segment_ids, attn_mask = x['input_ids'], x['segment_ids'], x['attention_mask']
        embedding_a = self(inputs=[input_ids, segment_ids, attn_mask], training=training)
        embedding_b = self(inputs=[input_ids, segment_ids, attn_mask], training=training)
        loss, y_pred = self._compute_contrastive_loss(embedding_a, embedding_b, y)
        return y_pred, loss

    def _compute_contrastive_loss(self, embedding_a, embedding_b, labels):
        norm_embedding_a = tf.linalg.normalize(embedding_a, axis=-1)[0]
        norm_embedding_b = tf.linalg.normalize(embedding_b, axis=-1)[0]
        # shape: (batch_size, batch_size)
        cosine = tf.matmul(norm_embedding_a, norm_embedding_b, transpose_b=True)
        # softmax temperature
        cosine = cosine / 0.05
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, cosine, from_logits=True)
        return loss, cosine


class SupervisedSimCSE(AbstractSimCSE):

    def forward(self, x, y, training=False):
        input_ids, segment_ids, attention_mask = x['input_ids'], x['segment_ids'], x['attention_mask']
        pos_input_ids, pos_segment_ids, pos_attention_mask = x['pos_input_ids'], x['pos_segment_ids'], x['pos_attention_mask']
        embedding = self(inputs=[input_ids, segment_ids, attention_mask], training=training)
        pos_embedding = self(inputs=[pos_input_ids, pos_segment_ids, pos_attention_mask], training=training)
        loss, y_pred = self._compute_contrastive_loss(embedding, pos_embedding, y)
        return y_pred, loss

    def _compute_contrastive_loss(self, embedding, pos_embedding, labels):
        norm_embedding = tf.linalg.normalize(embedding, axis=-1)[0]
        norm_pos_embedding = tf.linalg.normalize(pos_embedding, axis=-1)[0]
        cosine = tf.matmul(norm_embedding, norm_pos_embedding, transpose_b=True)
        # softmax temperature
        cosine = cosine / 0.05
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, cosine, from_logits=True)
        return loss, cosine


class HardNegativeSimCSE(AbstractSimCSE):

    def forward(self, x, y, training=False):
        input_ids, segment_ids, attention_mask = x['input_ids'], x['segment_ids'], x['attention_mask']
        pos_input_ids, pos_segment_ids, pos_attention_mask = x['pos_input_ids'], x['pos_segment_ids'], x['pos_attention_mask']
        neg_input_ids, neg_segment_ids, neg_attention_mask = x['neg_input_ids'], x['neg_segment_ids'], x['neg_attention_mask']
        embedding = self(inputs=[input_ids, segment_ids, attention_mask], training=training)
        pos_embedding = self(inputs=[pos_input_ids, pos_segment_ids, pos_attention_mask], training=training)
        neg_embedding = self(inputs=[neg_input_ids, neg_segment_ids, neg_attention_mask], training=training)
        loss, y_pred = self._compute_contrastive_loss(embedding, pos_embedding, neg_embedding, y)
        return y_pred, loss

    def _compute_contrastive_loss(self, embedding, pos_embedding, neg_embedding, labels):
        norm_embedding = tf.linalg.normalize(embedding, axis=-1)[0]
        norm_pos_embedding = tf.linalg.normalize(pos_embedding, axis=-1)[0]
        norm_neg_embedding = tf.linalg.normalize(neg_embedding, axis=-1)[0]
        # shape: (batch_size, batch_size)
        pos_sim = tf.matmul(norm_embedding, norm_pos_embedding, transpose_b=True)
        # shape: (batch_size, batch_size)
        neg_sim = tf.matmul(norm_embedding, norm_neg_embedding, transpose_b=True)
        # shape: (batch_size, batch_size * 2)
        cosine = tf.concat([pos_sim, neg_sim], axis=1)
        pos_weight = tf.zeros_like(pos_sim)
        # negative weight = 0.2
        neg_weight = tf.linalg.diag(tf.ones(tf.shape(neg_sim)[0])) * 0.2
        cosine_weight = tf.concat([pos_weight, neg_weight], axis=1)
        # softmax temperature
        cosine = (cosine + cosine_weight) / 0.05
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, cosine, from_logits=True)
        return loss, cosine
