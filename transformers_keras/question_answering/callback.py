import logging
from typing import List

import numpy as np
import tensorflow as tf
from smile_datasets import ExampleForQuestionAnswering, ParserForQuestionAnswering
from tokenizers import BertWordPieceTokenizer
from transformers_keras.common.metrics import ExactMatch, F1ForSequence

from . import readers


class BaseMetricForQuestionAnswering(tf.keras.callbacks.Callback):
    """Base metric for qa."""

    @classmethod
    def from_jsonl_files(
        cls,
        input_files,
        tokenizer: BertWordPieceTokenizer = None,
        vocab_file=None,
        context_key="context",
        question_key="question",
        do_lower_case=True,
        limit=None,
        **kwargs
    ):
        parser = ParserForQuestionAnswering(
            tokenizer=tokenizer, vocab_file=vocab_file, do_lower_case=do_lower_case, **kwargs
        )
        examples = []
        for instance in readers.read_jsonl_files_for_prediction(
            input_files, conetxt_key=context_key, question_key=question_key, **kwargs
        ):
            e = parser.parse(instance, **kwargs)
            if not e:
                continue
            examples.append(e)
            if limit is not None and len(examples) == limit:
                break
        return cls(examples, **kwargs)

    def __init__(self, examples: List[ExampleForQuestionAnswering], batch_size=32, **kwargs):
        super().__init__()
        self.examples = examples
        self.em = ExactMatch()
        self.batch_size = batch_size
        self.dataset = self._transform_examples_to_dataset(examples, **kwargs)

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

    def _transform_examples_to_dataset(self, examples, **kwargs) -> tf.data.Dataset:
        """transform examples to dataset"""

        def _to_dataset(x, dtype=tf.int32):
            x = tf.ragged.constant(x, dtype=dtype)
            d = tf.data.Dataset.from_tensor_slices(x)
            d = d.map(lambda x: x)
            return d

        dataset = tf.data.Dataset.zip(
            (
                _to_dataset(x=[e.input_ids for e in examples], dtype=tf.int32),
                _to_dataset(x=[e.segment_ids for e in examples], dtype=tf.int32),
                _to_dataset(x=[e.attention_mask for e in examples], dtype=tf.int32),
            )
        )
        pad_id = tf.constant(0, dtype=tf.int32)
        # fmt: off
        dataset = dataset.padded_batch(
            batch_size=self.batch_size,
            padded_shapes=([None,], [None,], [None,]),
            padding_values=(pad_id, pad_id, pad_id),
            drop_remainder=False,
        )
        # fmt: on
        dataset = dataset.map(lambda a, b, c: ({"input_ids": a, "segment_ids": b, "attention_mask": c}, None))
        return dataset


class EMForQuestionAnswering(BaseMetricForQuestionAnswering):
    """Exact Match metric for question answering."""

    def __init__(self, examples: List[ExampleForQuestionAnswering], batch_size=32, **kwargs):
        super().__init__(examples, batch_size=batch_size, **kwargs)

    def _compute_metric(self, gold_answers, pred_answers, epoch=0):
        acc = self.em(gold_answers, pred_answers, dim=1)
        tf.summary.scalar("EM", acc, step=epoch, description="EM Score")
        logging.info("No.%4d epoch EM: %.4f", epoch + 1, acc)


class F1ForQuestionAnswering(BaseMetricForQuestionAnswering):
    """F1 metric for question answering."""

    def __init__(self, examples: List[ExampleForQuestionAnswering], split_whitespace=False, **kwargs):
        super().__init__(examples, **kwargs)
        self.split_whitespace = split_whitespace
        self.f1 = F1ForSequence()

    def _compute_metric(self, gold_answers, pred_answers, epoch=0):
        score = self.f1(gold_answers, pred_answers, dim=1, split_whitespace=self.split_whitespace)
        tf.summary.scalar("F1", score, step=epoch, description="F1 Score")
        logging.info("No.%4d epoch F1: %.4f", epoch + 1, score)
