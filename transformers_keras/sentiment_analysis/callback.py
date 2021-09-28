import logging
from typing import List

import numpy as np
import tensorflow as tf
from transformers_keras.common.metrics import ExactMatch, F1ForSequence
from transformers_keras.sentiment_analysis.dataset import AspectTermExtractionDataset, AspectTermExtractionExample


class BaseMetricForAspectTermExtraction(tf.keras.callbacks.Callback):
    """Base metric for ATE"""

    @classmethod
    def from_jsonl_files(cls, input_files, limit=None, **kwargs):
        examples = AspectTermExtractionDataset.jsonl_to_examples(input_files, **kwargs)
        if limit is not None and limit > 0:
            examples = examples[:limit]
        return cls(examples, **kwargs)

    def __init__(self, examples: List[AspectTermExtractionExample], **kwargs):
        super().__init__()
        self.examples = examples
        self.dataset = AspectTermExtractionDataset.from_examples(self.examples, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        outputs = self.model.predict(self.dataset)
        all_start_ids, all_end_ids = outputs[0], outputs[1]
        pred_spans_list, gold_spans_list = [], []
        for start_ids, end_ids, example in zip(all_start_ids, all_end_ids, self.examples):
            start_ids, end_ids = np.argmax(start_ids, axis=-1), np.argmax(end_ids, axis=-1)
            pred_spans_list.append(self._decode_spans(start_ids.tolist(), end_ids.tolist(), example))
            gold_spans_list.append(self._decode_spans(example.start_ids, example.end_ids, example))
        self._compute_metric(gold_spans_list, pred_spans_list, epoch=epoch)

    def _compute_metric(self, gold_spans_list, pred_spans_list, epoch=0):
        pass

    def _decode_spans(self, start_ids, end_ids, example):
        spans = []
        span = []
        for idx, (start, end) in enumerate(zip(start_ids, end_ids)):
            if start == 1:
                span = [idx]
            if span and end == 1:
                span.append(idx)
                spans.append("".join([str(x).lstrip("##") for x in example.tokens[span[0] : span[1] + 1]]))
                span = []
        return spans


class ExactMatchForAspectTermExtraction(BaseMetricForAspectTermExtraction):
    """EM for ATE"""

    def __init__(self, examples: List[AspectTermExtractionExample], **kwargs) -> None:
        super().__init__(examples, **kwargs)
        self.em = ExactMatch()

    def _compute_metric(self, gold_spans_list, pred_spans_list, epoch=0):
        score = self.em(gold_spans_list, pred_spans_list, dim=2)
        logging.info("Epoch: %d, EM: %.6f", epoch, score)
        tf.summary.scalar("EM", score, description="EM score")


class F1ForAspectTermExtraction(BaseMetricForAspectTermExtraction):
    """F1 for ATE"""

    def __init__(self, examples: List[AspectTermExtractionExample], split_whitespace=False, **kwargs):
        super().__init__(examples, **kwargs)
        self.split_whitespace = split_whitespace
        self.f1 = F1ForSequence()

    def _compute_metric(self, gold_spans_list, pred_spans_list, epoch=0):
        score = self.f1(gold_spans_list, pred_spans_list, dim=2, split_whitespace=self.split_whitespace)
        logging.info("Epoch: %d, F1: %.6f", epoch, score)
        tf.summary.scalar("F1", score, description="F1 score")
