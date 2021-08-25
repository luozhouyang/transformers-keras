import collections
import logging
import re
import string
from typing import List

import numpy as np
import tensorflow as tf

from .dataset import QuestionAnsweringDataset, QuestionAnsweringExample


class BaseMetricForQuestionAnswering(tf.keras.callbacks.Callback):
    """Base metric for qa."""

    @classmethod
    def from_jsonl_files(cls, input_files, vocab_file, limit=None, **kwargs):
        examples = QuestionAnsweringDataset.jsonl_to_examples(input_files, vocab_file, **kwargs)
        if limit is not None and limit > 0:
            examples = examples[:limit]
        return cls(examples=examples, **kwargs)

    def __init__(self, examples: List[QuestionAnsweringExample], batch_size=32, **kwargs):
        super().__init__()
        self.examples = examples
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs):
        input_ids, segment_ids, attention_mask = self._build_inputs()
        dataset = tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensor_slices(input_ids),
                tf.data.Dataset.from_tensor_slices(segment_ids),
                tf.data.Dataset.from_tensor_slices(attention_mask),
            )
        ).batch(self.batch_size)
        dataset = dataset.map(lambda a, b, c: {"input_ids": a, "segment_ids": b, "attention_mask": c})
        outputs = self.model.predict(dataset)
        heads, tails = outputs[0], outputs[1]
        pred_answers, gold_answers = [], []
        for head, tail, example in zip(heads, tails, self.examples):
            head, tail = np.argmax(head), np.argmax(tail)
            pred_tokens = example.tokens[head : tail + 1]
            pred_text = "".join([str(x).lstrip("##") for x in pred_tokens])
            gold_text = "".join([str(x).lstrip("##") for x in example.tokens[example.start : example.end + 1]])
            pred_answers.append(self._normalize_answer(pred_text))
            gold_answers.append(self._normalize_answer(gold_text))
        self._compute_metric(epoch, pred_answers, gold_answers)

    def _compute_metric(self, epoch, pred_answers, gold_answers):
        raise NotImplementedError()

    def _build_inputs(self):
        input_ids, segment_ids, attention_mask = [], [], []
        maxlen = max([len(e.input_ids) for e in self.examples])
        for e in self.examples:
            _input_ids = e.input_ids + [0] * (maxlen - len(e.input_ids))
            _segment_ids = e.segment_ids + [0] * (maxlen - len(e.segment_ids))
            _attention_mask = e.attention_mask + [0] * (maxlen - len(e.attention_mask))
            input_ids.append(_input_ids)
            segment_ids.append(_segment_ids)
            attention_mask.append(_attention_mask)
        input_ids = tf.constant(input_ids, shape=(len(input_ids), maxlen), dtype=tf.int32)
        segment_ids = tf.constant(segment_ids, shape=(len(segment_ids), maxlen), dtype=tf.int32)
        attention_mask = tf.constant(attention_mask, shape=(len(attention_mask), maxlen), dtype=tf.int32)
        return input_ids, segment_ids, attention_mask

    def _normalize_answer(self, s):
        if not s:
            return ""
        # Lower text and remove punctuation, articles and extra whitespace.
        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))


class ExactMatchForQuestionAnswering(BaseMetricForQuestionAnswering):
    """Exact Match metric for question answering."""

    def _compute_metric(self, epoch, pred_answers, gold_answers):
        num_matchs = 0
        for pred, gold in zip(pred_answers, gold_answers):
            num_matchs += pred == gold
        acc = num_matchs * 1.0 / len(pred_answers)
        tf.summary.scalar("EM", acc, step=epoch, description="Exact Match Score")
        logging.info("No.%4d epoch EM: %.4f", epoch + 1, acc)


class F1ForQuestionAnswering(BaseMetricForQuestionAnswering):
    """F1 metric for question answering."""

    def __init__(self, examples: List[QuestionAnsweringExample], batch_size=32, split_whitespace=False, **kwargs):
        super().__init__(examples, batch_size, **kwargs)
        self.split_whitespace = split_whitespace

    def _compute_metric(self, epoch, pred_answers, gold_answers):
        f1_scores = []
        for pred, gold in zip(pred_answers, gold_answers):
            pred, gold = self._normalize_answer(pred), self._normalize_answer(gold)
            f1_scores.append(self._compute_f1(pred, gold))
        f1 = sum(f1_scores) / len(f1_scores)
        tf.summary.scalar("F1", f1, step=epoch, description="F1 Score")
        logging.info("No.%4d epoch F1: %.4f", epoch + 1, f1)

    def _compute_f1(self, pred, gold):
        pred_toks = pred.split()
        gold_toks = gold.split()
        if not self.split_whitespace:
            pred_toks = pred_toks[0] if pred_toks else ""
            gold_toks = gold_toks[0] if gold_toks else ""
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
