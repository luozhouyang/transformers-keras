import logging
import os

import numpy as np
import tensorflow as tf
from seqeval.metrics import classification_report
from smile_datasets import BertCharLevelTokenizer, LabelTokenizerForTokenClassification, ParserForTokenClassification

from . import readers


class SeqEvalForTokenClassification(tf.keras.callbacks.Callback):
    """Seqeval for token classification"""

    @classmethod
    def from_jsonl_files(
        cls,
        input_files,
        feature_tokenizer: BertCharLevelTokenizer = None,
        feature_vocab_file=None,
        label_tokenizer: LabelTokenizerForTokenClassification = None,
        label_vocab_file=None,
        feature_key="feature",
        label_key="label",
        **kwargs
    ):
        feature_tokenizer = feature_tokenizer or BertCharLevelTokenizer.from_file(feature_vocab_file, **kwargs)
        label_tokenizer = label_tokenizer or LabelTokenizerForTokenClassification.from_file(label_vocab_file, **kwargs)
        instances = []
        for instance in readers.read_jsonl_files_for_prediction(
            input_files, feature_key=feature_key, label_key=label_key, **kwargs
        ):
            instances.append(instance)
        return cls(instances, feature_tokenizer=feature_tokenizer, label_tokenizer=label_tokenizer, **kwargs)

    @classmethod
    def from_conll_files(
        cls,
        input_files,
        feature_tokenizer: BertCharLevelTokenizer = None,
        feature_vocab_file=None,
        label_tokenizer: LabelTokenizerForTokenClassification = None,
        label_vocab_file=None,
        sep="[\\s\t]+",
        **kwargs
    ):
        feature_tokenizer = feature_tokenizer or BertCharLevelTokenizer.from_file(feature_vocab_file, **kwargs)
        label_tokenizer = label_tokenizer or LabelTokenizerForTokenClassification.from_file(label_vocab_file, **kwargs)
        instances = []
        for instance in readers.read_conll_files_for_prediction(input_files, sep=sep, **kwargs):
            instances.append(instance)
        return cls(instances, feature_tokenizer=feature_tokenizer, label_tokenizer=label_tokenizer, **kwargs)

    def __init__(
        self,
        instances,
        feature_tokenizer: BertCharLevelTokenizer,
        label_tokenizer: LabelTokenizerForTokenClassification,
        batch_size=32,
        do_lower_case=True,
        o_token="O",
        **kwargs
    ):
        super().__init__()
        parser = ParserForTokenClassification.from_tokenizer(
            feature_tokenizer=feature_tokenizer,
            label_tokenizer=label_tokenizer,
            do_lower_case=do_lower_case,
            o_token=o_token,
            **kwargs
        )
        examples = []
        for instance in instances:
            e = parser.parse(instance, **kwargs)
            if not e:
                continue
            examples.append(e)
        self.examples = examples
        self.feature_tokenizer = feature_tokenizer
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
