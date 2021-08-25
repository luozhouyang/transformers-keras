import logging
import os
from typing import List

import numpy as np
import tensorflow as tf
from seqeval.metrics import classification_report
from transformers_keras.token_classification.dataset import TokenClassificationDataset, TokenClassificationExample
from transformers_keras.token_classification.tokenizer import TokenClassificationLabelTokenizer


class SeqEvalForTokenClassification(tf.keras.callbacks.Callback):
    """Seqeval for token classification"""

    @classmethod
    def from_conll_files(cls, input_files, vocab_file, label_vocab_file, sep="\t", **kwargs):
        examples = TokenClassificationDataset.conll_to_examples(
            input_files, vocab_file, label_vocab_file, sep=sep, **kwargs
        )
        label_tokenizer = TokenClassificationLabelTokenizer.from_file(
            label_vocab_file, o_token=kwargs.pop("o_token", "O")
        )
        return cls(examples=examples, label_tokenizer=label_tokenizer, **kwargs)

    def __init__(
        self,
        examples: List[TokenClassificationExample],
        label_tokenizer: TokenClassificationLabelTokenizer,
        batch_size=32,
        **kwargs
    ):
        super().__init__()
        self.examples = examples
        self.label_tokenizer = label_tokenizer
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs):
        dataset, labels = self._build_predict_dataset()
        pred_logits = self.model.predict(dataset)
        y_preds, y_trues = [], []
        for y_pred, y_true in zip(pred_logits, labels):
            y_pred = np.argmax(y_pred, axis=-1)
            y_pred = self.label_tokenizer.ids_to_labels(y_pred.tolist(), del_cls=False, del_sep=False)
            y_preds.append(y_pred)
            y_trues.append(y_true)
        report = classification_report(y_trues, y_preds)
        logging.info("SeqEval Reports at epoch {}:\n {}".format(epoch + 1, report))

    def _build_predict_dataset(self):
        input_ids, segment_ids, attention_mask, labels = self._build_inputs()
        dataset = tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensor_slices(input_ids),
                tf.data.Dataset.from_tensor_slices(segment_ids),
                tf.data.Dataset.from_tensor_slices(attention_mask),
            )
        ).batch(self.batch_size)
        dataset = dataset.map(lambda a, b, c: {"input_ids": a, "segment_ids": b, "attention_mask": c})
        return dataset, labels

    def _build_inputs(self):
        input_ids, segment_ids, attention_mask, labels = [], [], [], []
        maxlen = max([len(e.tokens) for e in self.examples])
        for e in self.examples:
            _input_ids = e.input_ids + [0] * (maxlen - len(e.input_ids))
            _segment_ids = e.segment_ids + [0] * (maxlen - len(e.segment_ids))
            _attention_mask = e.attention_mask + [0] * (maxlen - len(e.attention_mask))
            _labels = e.labels + ["O"] * (maxlen - len(e.labels))
            input_ids.append(_input_ids)
            segment_ids.append(_segment_ids)
            attention_mask.append(_attention_mask)
            labels.append(_labels)
        input_ids = tf.constant(input_ids, shape=(len(input_ids), maxlen), dtype=tf.int32)
        segment_ids = tf.constant(segment_ids, shape=(len(segment_ids), maxlen), dtype=tf.int32)
        attention_mask = tf.constant(attention_mask, shape=(len(attention_mask), maxlen), dtype=tf.int32)
        return input_ids, segment_ids, attention_mask, labels


class SeqEvalForCRFTokenClassification(SeqEvalForTokenClassification):
    """seq eval for CRF based models."""

    def on_epoch_end(self, epoch, logs):
        dataset, labels = self._build_predict_dataset()
        pred_ids = self.model.predict(dataset)
        y_preds, y_trues = [], []
        for y_pred, y_true in zip(pred_ids, labels):
            y_pred = self.label_tokenizer.ids_to_labels(y_pred.tolist(), del_cls=False, del_sep=False)
            y_preds.append(y_pred)
            y_trues.append(y_true)
        report = classification_report(y_trues, y_preds)
        logging.info("SeqEval Reports at epoch {}:\n {}".format(epoch + 1, report))

    def _build_predict_dataset(self):
        input_ids, segment_ids, attention_mask, labels = self._build_inputs()
        dataset = tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensor_slices(input_ids),
                tf.data.Dataset.from_tensor_slices(segment_ids),
                tf.data.Dataset.from_tensor_slices(attention_mask),
            )
        ).batch(self.batch_size)
        dataset = dataset.map(lambda a, b, c: ({"input_ids": a, "segment_ids": b, "attention_mask": c},))
        return dataset, labels


class SavedModelForCRFTokenClassification(tf.keras.callbacks.Callback):
    """Export model in SavedModel format for CRF based models."""

    def __init__(self, export_dir):
        super().__init__()
        self.export_dir = export_dir

    def on_epoch_end(self, epoch, logs):
        filepath = os.path.join(self.export_dir, "{epoch}").format(epoch=epoch + 1)
        self.model.save(filepath, signatures=self.model.forward, include_optimizer=False)
        logging.info("Saved model exported at: %s", filepath)
