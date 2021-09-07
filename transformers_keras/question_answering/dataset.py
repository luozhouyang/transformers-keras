import json
import logging
import re
from collections import namedtuple
from typing import List

import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
from transformers_keras.dataset_utils import AbstractDataset

QuestionAnsweringExample = namedtuple(
    "QuestionAnsweringExample", ["tokens", "input_ids", "segment_ids", "attention_mask", "start", "end"]
)
QuestionAnsweringXExample = namedtuple(
    "QuestionAnsweringXExample",
    ["tokens", "input_ids", "segment_ids", "attention_mask", "start", "end", "class_id"],
)


class QuestionAnsweringDataset(AbstractDataset):
    """Dataset builder for question answering models."""

    @classmethod
    def _build_dataset(
        cls,
        dataset,
        batch_size=64,
        repeat=None,
        max_sequence_length=512,
        bucket_boundaries=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        bucket_batch_sizes=None,
        buffer_size=1000000,
        seed=None,
        reshuffle_each_iteration=True,
        pad_id=0,
        auto_shard_policy=None,
        drop_remainder=False,
        **kwargs
    ):
        dataset = dataset.filter(
            lambda a, b, c, x, y: tf.size(a) <= max_sequence_length,
        )
        if repeat is not None:
            dataset = dataset.repeat(repeat)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)
        # fmt: off
        dataset = cls._bucketing(
            dataset,
            element_length_func=lambda a, b, c, x, y: tf.size(a),
            padded_shapes=([None, ], [None, ], [None, ], [], []),
            padding_values=(pad_id, pad_id, pad_id, None, None),
            batch_size=batch_size,
            pad_id=pad_id,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            drop_remainder=drop_remainder,
            **kwargs,
        )
        # fmt: on
        dataset = cls._to_dict(dataset)
        dataset = cls._auto_shard(dataset, auto_shard_policy=auto_shard_policy)
        return dataset

    @classmethod
    def _zip_dataset(cls, examples: List[QuestionAnsweringExample], **kwargs):
        """Zip datasets to one dataset."""

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
                _to_dataset(x=[e.start for e in examples], dtype=tf.int32),
                _to_dataset(x=[e.end for e in examples], dtype=tf.int32),
            )
        )
        return dataset

    @classmethod
    def _to_dict(cls, dataset):
        dataset = dataset.map(
            lambda a, b, c, x, y: ({"input_ids": a, "segment_ids": b, "attention_mask": c}, {"head": x, "tail": y}),
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        return dataset

    @classmethod
    def _parse_tfrecord(cls, dataset, **kwargs):
        features = {
            "input_ids": tf.io.VarLenFeature(tf.int64),
            "segment_ids": tf.io.VarLenFeature(tf.int64),
            "attention_mask": tf.io.VarLenFeature(tf.int64),
            "start": tf.io.VarLenFeature(tf.int64),
            "end": tf.io.VarLenFeature(tf.int64),
        }
        dataset = dataset.map(
            lambda x: tf.io.parse_example(x, features),
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        dataset = dataset.map(
            lambda x: (
                tf.cast(tf.sparse.to_dense(x["input_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["segment_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["attention_mask"]), tf.int32),
                tf.cast(tf.squeeze(tf.sparse.to_dense(x["start"])), tf.int32),
                tf.cast(tf.squeeze(tf.sparse.to_dense(x["end"])), tf.int32),
            ),
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        return dataset

    @classmethod
    def _example_to_tfrecord(cls, example: QuestionAnsweringExample, **kwargs):
        feature = {
            "input_ids": cls._int64_feature([int(x) for x in example.input_ids]),
            "segment_ids": cls._int64_feature([int(x) for x in example.segment_ids]),
            "attention_mask": cls._int64_feature([int(x) for x in example.attention_mask]),
            "start": cls._int64_feature([int(example.start)]),
            "end": cls._int64_feature([int(example.end)]),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    @classmethod
    def _parse_jsonl(cls, instances, tokenizer: BertWordPieceTokenizer = None, vocab_file=None, **kwargs):
        assert tokenizer or vocab_file, "`tokenizer` or `vocab_file` must be provided."
        if tokenizer is None:
            tokenizer = BertWordPieceTokenizer.from_file(
                vocab_file,
                lowercase=kwargs.get("do_lower_case", True),
            )
        examples = []
        for instance in instances:
            example = cls._instance_to_example(instance, tokenizer, **kwargs)
            if not example:
                continue
            examples.append(example)
        logging.info("Collected %d examples in total.", len(examples))
        return examples

    @classmethod
    def _instance_to_example(cls, instance, tokenizer: BertWordPieceTokenizer, **kwargs):
        context = instance[kwargs.get("context_key", "context")]
        question = instance[kwargs.get("question_key", "question")]
        answer = instance[kwargs.get("answer_key", "answer")]
        start_char_idx, end_char_idx = cls._find_answer_span(context, answer)
        if end_char_idx <= start_char_idx:
            return None
        # Mark the character indexes in context that are in answer
        is_char_in_ans = [0] * len(context)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1
        context_encoding = tokenizer.encode(context)
        # Find tokens that were created from answer characters
        ans_token_idx = []
        for idx, (start_char_idx, end_char_idx) in enumerate(context_encoding.offsets):
            if sum(is_char_in_ans[start_char_idx:end_char_idx]) > 0:
                ans_token_idx.append(idx)
        if not ans_token_idx:
            return None
        start_token_idx, end_token_idx = ans_token_idx[0], ans_token_idx[-1]
        question_encoding = tokenizer.encode(question)
        input_ids = context_encoding.ids + question_encoding.ids[1:]
        segment_ids = [0] * len(context_encoding.type_ids) + [1] * len(question_encoding.type_ids[1:])
        attention_mask = [1] * len(context_encoding.attention_mask + question_encoding.attention_mask[1:])
        assert len(input_ids) == len(segment_ids), "input_ids length:{} VS segment_ids length: {}".format(
            len(input_ids), len(segment_ids)
        )
        assert len(input_ids) == len(attention_mask), "input_ids length:{} VS attention_mask length: {}".format(
            len(input_ids), len(attention_mask)
        )

        example = QuestionAnsweringExample(
            tokens=context_encoding.tokens + question_encoding.tokens[1:],
            input_ids=input_ids,
            segment_ids=segment_ids,
            attention_mask=attention_mask,
            start=start_token_idx,
            end=end_token_idx,
        )
        return example

    @classmethod
    def _find_answer_span(cls, context, answer):
        for m in re.finditer(re.escape(answer), context, re.IGNORECASE):
            start, end = m.span()
            return start, end
        return 0, 0


class QuestionAnsweringXDataset(AbstractDataset):
    """Dataset builder for question answering models."""

    @classmethod
    def _build_dataset(
        cls,
        dataset,
        batch_size=64,
        repeat=None,
        max_sequence_length=512,
        bucket_boundaries=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        bucket_batch_sizes=None,
        buffer_size=1000000,
        seed=None,
        reshuffle_each_iteration=True,
        pad_id=0,
        auto_shard_policy=None,
        drop_remainder=False,
        **kwargs
    ):
        dataset = dataset.filter(
            lambda a, b, c, x, y, z: tf.size(a) <= max_sequence_length,
        )
        if repeat is not None:
            dataset = dataset.repeat(repeat)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)
        # fmt: off
        dataset = cls._bucketing(
            dataset,
            element_length_func=lambda a, b, c, x, y, z: tf.size(a),
            padded_shapes=([None, ], [None, ], [None, ], [], [], []),
            padding_values=(pad_id, pad_id, pad_id, None, None, None),
            batch_size=batch_size,
            pad_id=pad_id,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            drop_remainder=drop_remainder,
            **kwargs,
        )
        # fmt: on
        dataset = cls._to_dict(dataset)
        dataset = cls._auto_shard(dataset, auto_shard_policy=auto_shard_policy)
        return dataset

    @classmethod
    def _zip_dataset(cls, examples: List[QuestionAnsweringXExample], **kwargs):
        """Zip datasets to one dataset."""

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
                _to_dataset(x=[e.start for e in examples], dtype=tf.int32),
                _to_dataset(x=[e.end for e in examples], dtype=tf.int32),
                _to_dataset(x=[e.class_id for e in examples], dtype=tf.int32),
            )
        )
        return dataset

    @classmethod
    def _to_dict(cls, dataset, **kwargs):
        dataset = dataset.map(
            lambda a, b, c, x, y, z: (
                {"input_ids": a, "segment_ids": b, "attention_mask": c},
                {"head": x, "tail": y, "class": z},
            ),
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        return dataset

    @classmethod
    def _parse_tfrecord(cls, dataset, **kwargs):
        features = {
            "input_ids": tf.io.VarLenFeature(tf.int64),
            "segment_ids": tf.io.VarLenFeature(tf.int64),
            "attention_mask": tf.io.VarLenFeature(tf.int64),
            "start": tf.io.VarLenFeature(tf.int64),
            "end": tf.io.VarLenFeature(tf.int64),
            "class_id": tf.io.VarLenFeature(tf.int64),
        }
        dataset = dataset.map(
            lambda x: tf.io.parse_example(x, features),
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        dataset = dataset.map(
            lambda x: (
                tf.cast(tf.sparse.to_dense(x["input_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["segment_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["attention_mask"]), tf.int32),
                tf.cast(tf.squeeze(tf.sparse.to_dense(x["start"])), tf.int32),
                tf.cast(tf.squeeze(tf.sparse.to_dense(x["end"])), tf.int32),
                tf.cast(tf.squeeze(tf.sparse.to_dense(x["class_id"])), tf.int32),
            ),
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        return dataset

    @classmethod
    def _example_to_tfrecord(cls, example: QuestionAnsweringXExample, **kwargs):
        feature = {
            "input_ids": cls._int64_feature([int(x) for x in example.input_ids]),
            "segment_ids": cls._int64_feature([int(x) for x in example.segment_ids]),
            "attention_mask": cls._int64_feature([int(x) for x in example.attention_mask]),
            "start": cls._int64_feature([int(example.start)]),
            "end": cls._int64_feature([int(example.end)]),
            "class_id": cls._int64_feature([int(example.class_id)]),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    @classmethod
    def _parse_jsonl(cls, instances, tokenizer: BertWordPieceTokenizer = None, vocab_file=None, **kwargs):
        assert tokenizer or vocab_file, "`tokenizer` or `vocab_file` must be provided."
        if tokenizer is None:
            tokenizer = BertWordPieceTokenizer.from_file(
                vocab_file,
                lowercase=kwargs.get("do_lower_case", True),
            )
        examples = []
        for instance in instances:
            example = cls._instance_to_example(instance, tokenizer, **kwargs)
            if not example:
                continue
            examples.append(example)
        logging.info("Collected %d examples in total.", len(examples))
        return examples

    @classmethod
    def _instance_to_example(cls, instance, tokenizer: BertWordPieceTokenizer, **kwargs):
        context = instance[kwargs.get("context_key", "context")]
        question = instance[kwargs.get("question_key", "question")]
        answer = instance[kwargs.get("answer_key", "answer")]
        class_id = instance[kwargs.get("class_key", "class")]

        context_encoding = tokenizer.encode(context)
        question_encoding = tokenizer.encode(question)

        start_char_idx, end_char_idx = cls._find_answer_span(context, answer)
        # invalid answer index
        if end_char_idx <= start_char_idx:
            return cls._invalid_answer_example(instance, context_encoding, question_encoding, **kwargs)

        # Mark the character indexes in context that are in answer
        is_char_in_ans = [0] * len(context)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1
        # Find tokens that were created from answer characters
        ans_token_idx = []
        for idx, (start_char_idx, end_char_idx) in enumerate(context_encoding.offsets):
            if sum(is_char_in_ans[start_char_idx:end_char_idx]) > 0:
                ans_token_idx.append(idx)
        # set token index of start & end
        start_token_idx = ans_token_idx[0] if ans_token_idx else 0
        end_token_idx = ans_token_idx[-1] if ans_token_idx else 0
        # build example
        tokens = context_encoding.tokens + question_encoding.tokens[1:]
        input_ids = context_encoding.ids + question_encoding.ids[1:]
        segment_ids = [0] * len(context_encoding.type_ids) + [1] * len(context_encoding.type_ids[1:])
        attention_mask = [1] * len(context_encoding.attention_mask + question_encoding.attention_mask[1:])
        return QuestionAnsweringXExample(
            tokens=tokens,
            input_ids=input_ids,
            segment_ids=segment_ids,
            attention_mask=attention_mask,
            start=start_token_idx,
            end=end_token_idx,
            class_id=class_id,
        )

    @classmethod
    def _find_answer_span(cls, context, answer):
        for m in re.finditer(re.escape(answer), context, re.IGNORECASE):
            start, end = m.span()
            return start, end
        return 0, 0
