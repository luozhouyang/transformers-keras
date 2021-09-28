import logging
from typing import List

import numpy as np
import tensorflow as tf
from transformers_keras.common.metrics import ExactMatch, F1ForSequence

from .dataset import QuestionAnsweringDataset, QuestionAnsweringExample


class BaseMetricForQuestionAnswering(tf.keras.callbacks.Callback):
    """Base metric for qa."""

    @classmethod
    def from_jsonl_files(cls, input_files, vocab_file, limit=None, **kwargs):
        examples = QuestionAnsweringDataset.jsonl_to_examples(input_files, vocab_file=vocab_file, **kwargs)
        if limit is not None and limit > 0:
            examples = examples[:limit]
        return cls(examples, **kwargs)

    def __init__(self, examples, **kwargs):
        super().__init__()
        self.examples = examples
        self.dataset = QuestionAnsweringDataset.from_examples(self.examples, **kwargs)
        self.em = ExactMatch()

    def on_epoch_end(self, epoch, logs):
        outputs = self.model.predict(self.dataset)
        heads, tails = outputs[0], outputs[1]
        pred_answers, gold_answers = [], []
        for head, tail, example in zip(heads, tails, self.examples):
            head, tail = np.argmax(head), np.argmax(tail)
            pred_tokens = example.tokens[head : tail + 1]
            pred_text = "".join([str(x).lstrip("##") for x in pred_tokens])
            gold_text = "".join([str(x).lstrip("##") for x in example.tokens[example.start : example.end + 1]])
            pred_answers.append(pred_text)
            gold_answers.append(gold_text)
        self._compute_metric(gold_answers, pred_answers, epoch=epoch)

    def _compute_metric(self, gold_answers, pred_answers, epoch=0):
        raise NotImplementedError()


class ExactMatchForQuestionAnswering(BaseMetricForQuestionAnswering):
    """Exact Match metric for question answering."""

    def _compute_metric(self, gold_answers, pred_answers, epoch=0):
        acc = self.em(gold_answers, pred_answers, dim=1)
        tf.summary.scalar("EM", acc, step=epoch, description="EM Score")
        logging.info("No.%4d epoch EM: %.4f", epoch + 1, acc)


class F1ForQuestionAnswering(BaseMetricForQuestionAnswering):
    """F1 metric for question answering."""

    def __init__(self, examples: List[QuestionAnsweringExample], split_whitespace=False, **kwargs):
        super().__init__(examples, **kwargs)
        self.split_whitespace = split_whitespace
        self.f1 = F1ForSequence()

    def _compute_metric(self, gold_answers, pred_answers, epoch=0):
        score = self.f1(gold_answers, pred_answers, dim=1, split_whitespace=self.split_whitespace)
        tf.summary.scalar("F1", score, step=epoch, description="F1 Score")
        logging.info("No.%4d epoch F1: %.4f", epoch + 1, score)
