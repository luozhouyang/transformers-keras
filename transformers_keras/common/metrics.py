import collections
import logging

from . import utils


class ExactMatch:
    """Exact match"""

    def __call__(self, gold_spans, pred_spans, dim=1):
        if dim == 1:
            gold_spans_list, pred_spans_list = [gold_spans], [pred_spans]
        elif dim == 2:
            gold_spans_list, pred_spans_list = gold_spans, pred_spans
        else:
            raise ValueError("Invalid dim: " + str(dim))
        return self.call(gold_spans_list, pred_spans_list)

    def call(self, gold_spans_list, pred_spans_list):
        assert len(gold_spans_list) == len(pred_spans_list), "Length of inputs mismatch."
        matchs, total = 0, 0
        for gold_spans, pred_spans in zip(gold_spans_list, pred_spans_list):
            m, t = self._compute(gold_spans, pred_spans)
            matchs += m
            total += t
        return matchs * 1.0 / total

    def _compute(self, gold_spans, pred_spans):
        pred_spans = self._clip_pred_spans(gold_spans, pred_spans)
        num_matchs, num_total = 0, 0
        for idx in range(len(pred_spans)):
            pred = utils.normalize_span(pred_spans[idx])
            gold = utils.normalize_span(gold_spans[idx])
            num_matchs += pred == gold
        num_total += len(gold_spans)
        return num_matchs, num_total

    def _clip_pred_spans(self, gold_spans, pred_spans):
        if len(pred_spans) > len(gold_spans):
            # logging.warning("len(pred_spans) > len(gold_spans), clipping pred_spans...")
            pred_spans = pred_spans[: len(gold_spans)]
        return pred_spans


class F1ForSequence:
    """F1 for sequence tasks."""

    def __call__(self, gold_spans, pred_spans, dim=1, split_whitespace=False):
        if dim == 1:
            gold_spans_list, pred_spans_list = [gold_spans], [pred_spans]
        elif dim == 2:
            gold_spans_list, pred_spans_list = gold_spans, pred_spans
        else:
            raise ValueError("Invalid dim: " + str(dim))
        return self.call(gold_spans_list, pred_spans_list, split_whitespace=split_whitespace)

    def call(self, gold_spans_list, pred_spans_list, split_whitespace=False):
        assert len(gold_spans_list) == len(pred_spans_list), "Length of inputs mismatch."
        scores = []
        for gold_spans, pred_spans in zip(gold_spans_list, pred_spans_list):
            scores.extend(self._compute_f1(gold_spans, pred_spans, split_whitespace=split_whitespace))
        return sum(scores) / len(scores)

    def _compute_f1(self, gold_spans, pred_spans, split_whitespace=False):
        if len(pred_spans) > len(gold_spans):
            # logging.warning("len(pred_spans) > len(gold_spans), clipping pred_spans...")
            pred_spans = pred_spans[: len(gold_spans)]
        scores = []
        for idx in range(len(pred_spans)):
            gold = utils.normalize_span(gold_spans[idx])
            pred = utils.normalize_span(pred_spans[idx])
            scores.append(self._compute(gold, pred, split_whitespace=split_whitespace))
        return scores

    def _compute(self, pred, gold, split_whitespace=False):
        pred_toks = pred.split()
        gold_toks = gold.split()
        if not split_whitespace:
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
