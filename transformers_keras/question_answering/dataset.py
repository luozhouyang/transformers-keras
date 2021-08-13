import json
import logging
import os
from collections import namedtuple
from typing import List

import tensorflow as tf

QuestionAnsweringExample = namedtuple(
    "QuestionAnsweringExample", ["text", "tokens", "input_ids", "segment_ids", "attention_mask", "start", "end"]
)


class QuestionAnsweringDataset:
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
            pad_id=0,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            drop_remainder=drop_remainder,
            **kwargs,
        )
        dataset = cls._to_dict(dataset)
        dataset = cls._auto_shard(dataset, auto_shard_policy=auto_shard_policy)
        return dataset

    @classmethod
    def from_jsonl_files(
        cls,
        input_files,
        fn,
        batch_size=64,
        repeat=None,
        max_sequence_length=512,
        bucket_boundaries=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        bucket_batch_sizes=None,
        buffer_size=1000000,
        seed=None,
        reshuffle_each_iteration=True,
        drop_remainder=False,
        pad_id=0,
        auto_shard_policy=None,
        **kwargs
    ):
        """Build dataset from jsonl files.

        Args:
            input_files: File path or list of file paths of jsonl files
            fn: A callable to parse line to examples.
                It's signature is: def fn(instance, **kwargs) -> List[QuestionAnsweringExample],
                in which instance is a dict load from each line.
            batch_size: Batch size of dataset
            max_sequence_length: Max sequence length of inputs. If sequence is longer than this, it will be filter out.
            bucket_boundaries: List of integer, used to group sequence to different buckets according it's length
            buffer_size: Python integer, buffer size for dataset, should greater equal than total number of training examples
            seed: Python integer, seed for shuffling examples
            reshuffle_each_iteration: Python boolean, reshuffle examples when epoch ends or not
            pad_id: Python integer, pad id to sequence, default to 0
            auto_shard_policy: Auto shard policy, used in distributed training
        """
        examples = cls._read_jsonl_examples(
            input_files=input_files,
            fn=fn,
            max_sequence_length=max_sequence_length,
            **kwargs,
        )
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
            **kwargs,
        )

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
        **kwargs
    ):
        logging.info("Load %d examples in total.", len(examples))
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
    def _read_jsonl_examples(cls, input_files, fn, **kwargs) -> List[QuestionAnsweringExample]:
        all_examples = []
        if isinstance(input_files, str):
            input_files = [input_files]
        for f in input_files:
            if not os.path.exists(f):
                logging.warning("File: %s does not exist, skipped.", f)
                continue
            with open(f, mode="rt", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    instance = json.loads(line)
                    examples = fn(instance, **kwargs)
                    all_examples.extend(examples)
        return all_examples

    @classmethod
    def _read_tfrecord(cls, input_files, **kwargs):
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
                tf.cast(tf.squeeze(tf.sparse.to_dense(x["end"])) - 1, tf.int32),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)
        return dataset
