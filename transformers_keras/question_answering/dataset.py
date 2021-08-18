import json
import logging
import re
from collections import namedtuple
from typing import List

import tensorflow as tf
from tokenizers import BertWordPieceTokenizer

QuestionAnsweringExample = namedtuple(
    "QuestionAnsweringExample", ["text", "tokens", "input_ids", "segment_ids", "attention_mask", "start", "end"]
)
QuestionAnsweringXExample = namedtuple(
    "QuestionAnsweringXExample",
    ["text", "tokens", "input_ids", "segment_ids", "attention_mask", "start", "end", "class_id"],
)


class _BaseQuestionAnsweringDataset:
    """Dataset transformation"""

    @classmethod
    def from_jsonl_files(
        cls,
        input_files,
        vocab_file,
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
        verbose=True,
        **kwargs
    ):
        examples = cls.jsonl_to_examples(input_files, vocab_file, **kwargs)
        return cls.from_examples(
            examples,
            batch_size=batch_size,
            repeat=repeat,
            max_sequence_length=max_sequence_length,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            buffer_size=buffer_size,
            seed=seed,
            reshuffle_each_iteration=reshuffle_each_iteration,
            pad_id=pad_id,
            auto_shard_policy=auto_shard_policy,
            drop_remainder=drop_remainder,
            verbose=verbose,
            **kwargs,
        )

    @classmethod
    def from_examples(
        cls,
        examples,
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
        verbose=True,
        **kwargs
    ):
        raise NotImplementedError()

    @classmethod
    def jsonl_to_tfrecord(cls, input_files, vocab_file, output_files, **kwargs):
        examples = cls.jsonl_to_examples(input_files, vocab_file, **kwargs)
        cls.examples_to_tfrecord(examples, output_files, **kwargs)

    @classmethod
    def jsonl_to_examples(cls, input_files, vocab_file, **kwargs):
        if isinstance(input_files, str):
            input_files = [input_files]
        instances = []
        for f in input_files:
            with open(f, mode="rt", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    instance = json.loads(line)
                    instances.append(instance)
        logging.info("Collected %d instances in total.", len(instances))
        examples = []
        tokenizer = BertWordPieceTokenizer.from_file(vocab_file, lowercase=kwargs.pop("do_lower_case", True))
        for instance in instances:
            example = cls._instance_to_example(instance, tokenizer, **kwargs)
            if not example:
                continue
            examples.append(example)
        logging.info("Collected %d examples in total.", len(examples))
        return examples

    @classmethod
    def _instance_to_example(cls, instance, tokenizer: BertWordPieceTokenizer, **kwargs):
        raise NotImplementedError()

    @classmethod
    def examples_to_tfrecord(cls, examples, output_files, **kwargs):
        if isinstance(output_files, str):
            output_files = [output_files]
        writers = [tf.io.TFRecordWriter(f) for f in output_files]
        idx = 0
        for example in examples:
            tfrecord_example = cls._example_to_tfrecord(example)
            writers[idx].write(tfrecord_example.SerializeToString())
            idx += 1
            idx = idx % len(writers)
        for w in writers:
            w.close()
        logging.info("Finished to write %d examples to tfrecords.", len(examples))

    @classmethod
    def _example_to_tfrecord(cls, example, **kwargs):
        raise NotImplementedError()

    @classmethod
    def _find_answer_span(cls, context, answer):
        for m in re.finditer(re.escape(answer), context, re.IGNORECASE):
            start, end = m.span()
            return start, end
        return 0, 0

    @classmethod
    def _int64_feature(cls, values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


class QuestionAnsweringDataset(_BaseQuestionAnsweringDataset):
    """Dataset builder for question answering models."""

    @classmethod
    def from_tfrecord_files(
        cls,
        input_files,
        batch_size=64,
        repeat=None,
        max_sequence_length=512,
        bucket_boundaries=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        bucket_batch_sizes=None,
        buffer_size=1000000,
        seed=None,
        reshuffle_each_iteration=True,
        pad_id=0,
        drop_remainder=False,
        auto_shard_policy=None,
        **kwargs
    ):
        dataset = cls._read_tfrecord(input_files, **kwargs)
        dataset = dataset.filter(lambda a, b, c, x, y: tf.size(a) <= max_sequence_length)
        if repeat is not None:
            dataset = dataset.repeat(repeat)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)
        dataset = cls._bucketing(
            dataset,
            batch_size=batch_size,
            pad_id=pad_id,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            drop_remainder=drop_remainder,
            **kwargs,
        )
        dataset = cls._to_dict(dataset)
        dataset = cls._auto_shard(dataset, auto_shard_policy=auto_shard_policy)
        return dataset

    @classmethod
    def from_examples(
        cls,
        examples: List[QuestionAnsweringExample],
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
        verbose=True,
        **kwargs
    ):
        logging.info("Load %d examples in total.", len(examples))
        if verbose:
            cls._show_examples(examples, n=5, **kwargs)
        dataset = cls._zip_dataset(examples)
        dataset = dataset.filter(
            lambda a, b, c, x, y: tf.size(a) <= max_sequence_length,
        )
        if repeat is not None:
            dataset = dataset.repeat(repeat)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)
        dataset = cls._bucketing(
            dataset,
            batch_size=batch_size,
            pad_id=pad_id,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            drop_remainder=drop_remainder,
            **kwargs,
        )
        dataset = cls._to_dict(dataset)
        dataset = cls._auto_shard(dataset, auto_shard_policy=auto_shard_policy)
        return dataset

    @classmethod
    def _show_examples(cls, examples, n=5, **kwargs):
        n = min(n, len(examples))
        logging.info("Showing %d examples.", n)
        for i in range(n):
            logging.info("NO.%d example: %s", i, examples[i])

    @classmethod
    def _zip_dataset(cls, examples: List[QuestionAnsweringExample]):
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
    def _bucketing(
        cls,
        dataset,
        batch_size=64,
        pad_id=0,
        bucket_boundaries=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        bucket_batch_sizes=None,
        drop_remainder=False,
        **kwargs
    ):
        if bucket_batch_sizes is None:
            bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
        assert (
            len(bucket_batch_sizes) == len(bucket_boundaries) + 1
        ), "len(bucket_batch_sizes) should equals len(bucket_doundaries) + 1"

        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
            element_length_func=lambda a, b, c, x, y: tf.size(a),
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            padded_shapes=([None, ], [None, ], [None, ], [], []),
            padding_values=(pad_id, pad_id, pad_id, None, None),
            drop_remainder=drop_remainder,
        )).prefetch(tf.data.AUTOTUNE)
        # fmt: on
        return dataset

    @classmethod
    def _to_dict(cls, dataset):
        dataset = dataset.map(
            lambda a, b, c, x, y: ({"input_ids": a, "segment_ids": b, "attention_mask": c}, {"head": x, "tail": y}),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)
        return dataset

    @classmethod
    def _auto_shard(cls, dataset, auto_shard_policy=None):
        if auto_shard_policy is not None:
            options = tf.data.Options()
            # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            options.experimental_distribute.auto_shard_policy = auto_shard_policy
            dataset = dataset.with_options(options)
        return dataset

    @classmethod
    def _read_tfrecord(cls, input_files, **kwargs):
        if isinstance(input_files, str):
            input_files = [input_files]
        if len(input_files) == 1:
            dataset = tf.data.TFRecordDataset(input_files)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(input_files)
            dataset = dataset.interleave(
                lambda x: tf.data.TFRecordDataset(x),
                cycle_length=len(input_files),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        features = {
            "input_ids": tf.io.VarLenFeature(tf.int64),
            "segment_ids": tf.io.VarLenFeature(tf.int64),
            "attention_mask": tf.io.VarLenFeature(tf.int64),
            "start": tf.io.VarLenFeature(tf.int64),
            "end": tf.io.VarLenFeature(tf.int64),
        }
        dataset = dataset.map(
            lambda x: tf.io.parse_example(x, features),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)
        dataset = dataset.map(
            lambda x: (
                tf.cast(tf.sparse.to_dense(x["input_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["segment_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["attention_mask"]), tf.int32),
                tf.cast(tf.squeeze(tf.sparse.to_dense(x["start"])), tf.int32),
                tf.cast(tf.squeeze(tf.sparse.to_dense(x["end"])), tf.int32),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)
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
        example = QuestionAnsweringExample(
            text=context + question,
            tokens=context_encoding.tokens + question_encoding.tokens[1:],
            input_ids=context_encoding.ids + question_encoding.ids[1:],
            segment_ids=[0] * len(context_encoding.type_ids) + [1] * len(context_encoding.type_ids[1:]),
            attention_mask=[1] * len(context_encoding.attention_mask + question_encoding.attention_mask[1:]),
            start=start_token_idx,
            end=end_token_idx,
        )
        return example


class QuestionAnsweringXDataset(_BaseQuestionAnsweringDataset):
    """Dataset builder for question answering models."""

    @classmethod
    def from_tfrecord_files(
        cls,
        input_files,
        batch_size=64,
        repeat=None,
        max_sequence_length=512,
        bucket_boundaries=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        bucket_batch_sizes=None,
        buffer_size=1000000,
        seed=None,
        reshuffle_each_iteration=True,
        pad_id=0,
        drop_remainder=False,
        auto_shard_policy=None,
        **kwargs
    ):
        dataset = cls._read_tfrecord(input_files, **kwargs)
        dataset = dataset.filter(lambda a, b, c, x, y, z: tf.size(a) <= max_sequence_length)
        if repeat is not None:
            dataset = dataset.repeat(repeat)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)
        dataset = cls._bucketing(
            dataset,
            batch_size=batch_size,
            pad_id=pad_id,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            drop_remainder=drop_remainder,
            **kwargs,
        )
        dataset = cls._to_dict(dataset)
        dataset = cls._auto_shard(dataset, auto_shard_policy=auto_shard_policy)
        return dataset

    @classmethod
    def from_examples(
        cls,
        examples: List[QuestionAnsweringXExample],
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
        verbose=True,
        **kwargs
    ):
        logging.info("Load %d examples in total.", len(examples))
        if verbose:
            cls._show_examples(examples, n=5, **kwargs)
        dataset = cls._zip_dataset(examples)
        dataset = dataset.filter(
            lambda a, b, c, x, y, z: tf.size(a) <= max_sequence_length,
        )
        if repeat is not None:
            dataset = dataset.repeat(repeat)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)
        dataset = cls._bucketing(
            dataset,
            batch_size=batch_size,
            pad_id=pad_id,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            drop_remainder=drop_remainder,
            **kwargs,
        )
        dataset = cls._to_dict(dataset)
        dataset = cls._auto_shard(dataset, auto_shard_policy=auto_shard_policy)
        return dataset

    @classmethod
    def _show_examples(cls, examples, n=5, **kwargs):
        n = min(n, len(examples))
        logging.info("Showing %d examples.", n)
        for i in range(n):
            logging.info("NO.%d example: %s", i, examples[i])

    @classmethod
    def _zip_dataset(cls, examples: List[QuestionAnsweringXExample]):
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
    def _bucketing(
        cls,
        dataset,
        batch_size=64,
        pad_id=0,
        bucket_boundaries=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        bucket_batch_sizes=None,
        drop_remainder=False,
        **kwargs
    ):
        if bucket_batch_sizes is None:
            bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
        assert (
            len(bucket_batch_sizes) == len(bucket_boundaries) + 1
        ), "len(bucket_batch_sizes) should equals len(bucket_doundaries) + 1"

        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
            element_length_func=lambda a, b, c, x, y, z: tf.size(a),
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            padded_shapes=([None, ], [None, ], [None, ], [], [], []),
            padding_values=(pad_id, pad_id, pad_id, None, None, None),
            drop_remainder=drop_remainder,
        )).prefetch(tf.data.AUTOTUNE)
        # fmt: on
        return dataset

    @classmethod
    def _to_dict(cls, dataset):
        dataset = dataset.map(
            lambda a, b, c, x, y, z: (
                {"input_ids": a, "segment_ids": b, "attention_mask": c},
                {"head": x, "tail": y, "class": z},
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)
        return dataset

    @classmethod
    def _auto_shard(cls, dataset, auto_shard_policy=None):
        if auto_shard_policy is not None:
            options = tf.data.Options()
            # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            options.experimental_distribute.auto_shard_policy = auto_shard_policy
            dataset = dataset.with_options(options)
        return dataset

    @classmethod
    def _read_tfrecord(cls, input_files, **kwargs):
        if isinstance(input_files, str):
            input_files = [input_files]
        if len(input_files) == 1:
            dataset = tf.data.TFRecordDataset(input_files)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(input_files)
            dataset = dataset.interleave(
                lambda x: tf.data.TFRecordDataset(x),
                cycle_length=len(input_files),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
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
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)
        dataset = dataset.map(
            lambda x: (
                tf.cast(tf.sparse.to_dense(x["input_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["segment_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["attention_mask"]), tf.int32),
                tf.cast(tf.squeeze(tf.sparse.to_dense(x["start"])), tf.int32),
                tf.cast(tf.squeeze(tf.sparse.to_dense(x["end"])), tf.int32),
                tf.cast(tf.squeeze(tf.sparse.to_dense(x["class_id"])), tf.int32),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)
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
    def _instance_to_example(cls, instance, tokenizer: BertWordPieceTokenizer, **kwargs):
        context = instance[kwargs.get("context_key", "context")]
        question = instance[kwargs.get("question_key", "question")]
        answer = instance[kwargs.get("answer_key", "answer")]
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
        # no answer tokens found
        if not ans_token_idx:
            return cls._invalid_answer_example(instance, context_encoding, question_encoding, **kwargs)

        start_token_idx, end_token_idx = ans_token_idx[0], ans_token_idx[-1]
        return cls._valid_answer_example(
            instance, context_encoding, question_encoding, start_token_idx, end_token_idx, **kwargs
        )

    @classmethod
    def _invalid_answer_example(cls, instance, context_encoding, question_encoding, **kwargs):
        context = instance[kwargs.get("context_key", "context")]
        question = instance[kwargs.get("question_key", "question")]
        class_id = instance[kwargs.get("class_key", "class")]
        # build inputs
        tokens = context_encoding.tokens + question_encoding.tokens[1:]
        input_ids = context_encoding.ids + question_encoding.ids[1:]
        segment_ids = [0] * len(context_encoding.type_ids) + [1] * len(context_encoding.type_ids[1:])
        attention_mask = [1] * len(context_encoding.attention_mask + question_encoding.attention_mask[1:])
        return QuestionAnsweringXExample(
            text=context + question,
            tokens=tokens,
            input_ids=input_ids,
            segment_ids=segment_ids,
            attention_mask=attention_mask,
            start=0,
            end=0,
            class_id=class_id,
        )

    @classmethod
    def _valid_answer_example(
        cls, instance, context_encoding, question_encoding, start_token_idx, end_token_idx, **kwargs
    ):
        context = instance[kwargs.get("context_key", "context")]
        question = instance[kwargs.get("question_key", "question")]
        class_id = instance[kwargs.get("class_key", "class")]
        # build inputs
        tokens = context_encoding.tokens + question_encoding.tokens[1:]
        input_ids = context_encoding.ids + question_encoding.ids[1:]
        segment_ids = [0] * len(context_encoding.type_ids) + [1] * len(context_encoding.type_ids[1:])
        attention_mask = [1] * len(context_encoding.attention_mask + question_encoding.attention_mask[1:])
        return QuestionAnsweringXExample(
            text=context + question,
            tokens=tokens,
            input_ids=input_ids,
            segment_ids=segment_ids,
            attention_mask=attention_mask,
            start=start_token_idx,
            end=end_token_idx,
            class_id=class_id,
        )
