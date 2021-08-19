import json
import logging
from collections import namedtuple
from typing import List

import scipy.stats
import tensorflow as tf
from tensorflow import keras
from tokenizers import BertWordPieceTokenizer

ExampleForSpearman = namedtuple("ExampleForSpearman", ["sentence_a", "sentence_b", "label"])


class SpearmanForSentenceEmbedding(tf.keras.callbacks.Callback):
    """Spearman correlation for sentence embedding."""

    @classmethod
    def from_jsonl_files(cls, input_files, vocab_file, **kwargs):
        if isinstance(input_files, str):
            input_files = [input_files]
        examples = []
        for f in input_files:
            with open(f, mode="rt", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    instance = json.loads(line)
                    sentence_a = instance[kwargs.pop("sentence_a_key", "sentence_a")]
                    sentence_b = instance[kwargs.pop("sentence_b_key", "sentence_b")]
                    label = instance[keras.pop("label_key", "label")]
                    examples.append(
                        ExampleForSpearman(
                            sentence_a=sentence_a,
                            sentence_b=sentence_b,
                            label=label,
                        )
                    )
        tokenizer = BertWordPieceTokenizer.from_file(vocab_file, lowercase=kwargs.pop("do_lower_case", True))
        return cls(examples=examples, tokenzer=tokenizer, **kwargs)

    def __init__(self, examples: List[ExampleForSpearman], tokenizer: BertWordPieceTokenizer, batch_size=32, **kwargs):
        super().__init__()
        self.examples = examples
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs):
        a_sentences = [x.sentence_a for x in self.examples]
        b_sentences = [x.sentence_b for x in self.examples]
        labels = [x.label for x in self.examples]
        a_dataset = self._build_inputs(a_sentences)
        b_dataset = self._build_inputs(b_sentences)
        a_embeddings = self.model.predict(a_dataset)
        b_embeddings = self.model.predict(b_dataset)
        correlation = self._report(a_embeddings, b_embeddings, labels)
        logging.info("No.%4d epoch Spearman Correlation: %.4f", epoch + 1, correlation)
        tf.summary.scalar("SpearmanCorrelation", correlation, step=epoch)

    def _report(self, a_embeddings, b_embeddings, labels):
        y_pred, y_true = [], []
        for a, b, l in zip(a_embeddings, b_embeddings, labels):
            norm_a = tf.linalg.normalize(a, axis=-1)[0]
            norm_b = tf.linalg.normalize(b, axis=-1)[0]
            sim = tf.reduce_sum(norm_a * norm_b, axis=-1)
            y_pred.append(sim.numpy())
            y_true.append(l)
        spearman = scipy.stats.spearmanr(y_pred, y_true)
        # print('spearman: ', spearman)
        return spearman.correlation

    def _build_inputs(self, sentences):
        input_ids, segment_ids, attention_mask = [], [], []
        for sent in sentences:
            encoding = self.tokenizer.encode(sent)
            input_ids.append(encoding.ids)
            segment_ids.append(encoding.type_ids)
            attention_mask.append(encoding.attention_mask)
        padded_input_ids, padded_segment_ids, padded_attention_mask = [], [], []
        maxlen = max([len(x) for x in input_ids])
        for ids, type_ids, mask in zip(input_ids, segment_ids, attention_mask):
            padded_input_ids.append(ids + [0] * (maxlen - len(ids)))
            padded_segment_ids.append(type_ids + [0] * (maxlen - len(type_ids)))
            padded_attention_mask.append(mask + [0] * (maxlen - len(mask)))

        input_ids = tf.constant(padded_input_ids, shape=(len(padded_input_ids), maxlen), dtype=tf.int32)
        segment_ids = tf.constant(padded_segment_ids, shape=(len(padded_segment_ids), maxlen), dtype=tf.int32)
        attention_mask = tf.constant(padded_attention_mask, shape=(len(padded_attention_mask), maxlen), dtype=tf.int32)
        dataset = tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensor_slices(input_ids),
                tf.data.Dataset.from_tensor_slices(segment_ids),
                tf.data.Dataset.from_tensor_slices(attention_mask),
            )
        )
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(lambda a, b, c: {"input_ids": a, "segment_ids": b, "attention_mask": c})
        return dataset
