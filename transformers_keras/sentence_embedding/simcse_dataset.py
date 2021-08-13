import json
import logging
import os
import re
from collections import namedtuple
from typing import List

import tensorflow as tf

SimCSEExample = namedtuple(
    "SimCSEExample",
    [
        "sequence",
        "input_ids",
        "segment_ids",
        "attention_mask",
        "pos_sequence",
        "pos_input_ids",
        "pos_segment_ids",
        "pos_attention_mask",
        "neg_sequence",
        "neg_input_ids",
        "neg_segment_ids",
        "neg_attention_mask",
    ],
)


class SimCSEDataset:
    """Dataset builder for SimCSE models."""

    @classmethod
    def from_tfrecord_files(
        cls,
        input_files,
        batch_size=64,
        repeat=None,
        with_pos_sequence=False,
        with_neg_sequence=False,
        max_sequence_length=512,
        bucket_boundaries=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        bucket_batch_sizes=None,
        buffer_size=1000000,
        seed=None,
        reshuffle_each_iteration=True,
        pad_id=0,
        auto_shard_policy=None,
        drop_remainder=True,
        **kwargs
    ):
        dataset = cls._read_tfrecord(
            input_files, with_pos_sequence=with_pos_sequence, with_neg_sequence=with_neg_sequence, **kwargs
        )
        return cls._build(
            dataset,
            batch_size=batch_size,
            repeat=repeat,
            with_pos_sequence=with_pos_sequence,
            with_neg_sequence=with_neg_sequence,
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
    def from_jsonl_files(
        cls,
        input_files,
        fn,
        batch_size=64,
        repeat=None,
        with_pos_sequence=False,
        with_neg_sequence=False,
        max_sequence_length=512,
        bucket_boundaries=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        bucket_batch_sizes=None,
        buffer_size=1000000,
        seed=None,
        reshuffle_each_iteration=True,
        pad_id=0,
        auto_shard_policy=None,
        drop_remainder=True,
        verbose=True,
        **kwargs
    ):
        examples = cls._read_jsonl_examples(
            input_files, fn=fn, with_pos_sequence=with_pos_sequence, with_neg_sequence=with_neg_sequence, **kwargs
        )
        return cls.from_examples(
            examples,
            batch_size=batch_size,
            repeat=repeat,
            with_pos_sequence=with_pos_sequence,
            with_neg_sequence=with_neg_sequence,
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
        examples: List[SimCSEExample],
        with_pos_sequence=False,
        with_neg_sequence=False,
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
        drop_remainder=True,
        verbose=True,
        **kwargs
    ):
        logging.info("Number of examples: %d", len(examples))
        cls._show_examples(examples, n=5, verbose=verbose, **kwargs)
        dataset = cls._zip_dataset(
            examples, with_pos_sequence=with_pos_sequence, with_neg_sequence=with_neg_sequence, **kwargs
        )
        return cls._build(
            dataset,
            batch_size=batch_size,
            repeat=repeat,
            with_pos_sequence=with_pos_sequence,
            with_neg_sequence=with_neg_sequence,
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
    def _build(
        cls,
        dataset,
        with_pos_sequence=False,
        with_neg_sequence=False,
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
        drop_remainder=True,
        **kwargs
    ):
        dataset = cls._filter_dataset(
            dataset,
            max_sequence_length=max_sequence_length,
            with_pos_sequence=with_pos_sequence,
            with_neg_sequence=with_neg_sequence,
            **kwargs,
        )
        if repeat is not None:
            dataset = dataset.repeat(repeat)
        dataset = dataset.shuffle(
            buffer_size=buffer_size,
            seed=seed,
            reshuffle_each_iteration=reshuffle_each_iteration,
        )
        dataset = cls._bucketing(
            dataset,
            with_pos_sequence=with_pos_sequence,
            with_neg_sequence=with_neg_sequence,
            batch_size=batch_size,
            pad_id=pad_id,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            drop_remainder=drop_remainder,
            **kwargs,
        )
        dataset = cls._to_dict(
            dataset, with_pos_sequence=with_pos_sequence, with_neg_sequence=with_neg_sequence, **kwargs
        )
        dataset = cls._auto_shard(dataset, auto_shard_policy=auto_shard_policy)
        return dataset

    @classmethod
    def _show_examples(cls, examples, n=5, verbose=True, **kwargs):
        if not verbose:
            return
        n = min(n, len(examples))
        logging.info("Showing %d examples.", n)
        for i in range(n):
            logging.info("No.%d example: %s", i, examples[i])

    @classmethod
    def _zip_dataset(cls, examples: List[SimCSEExample], with_pos_sequence=False, with_neg_sequence=False, **kwargs):
        """zip dataset"""

        def _to_dataset(x, dtype=tf.int32):
            x = tf.ragged.constant(x, dtype=dtype)
            d = tf.data.Dataset.from_tensor_slices(x)
            d = d.map(lambda x: x)
            return d

        if with_neg_sequence:
            dataset = tf.data.Dataset.zip(
                (
                    _to_dataset([e.input_ids for e in examples]),
                    _to_dataset([e.segment_ids for e in examples]),
                    _to_dataset([e.attention_mask for e in examples]),
                    _to_dataset([e.pos_input_ids for e in examples]),
                    _to_dataset([e.pos_segment_ids for e in examples]),
                    _to_dataset([e.pos_attention_mask for e in examples]),
                    _to_dataset([e.neg_input_ids for e in examples]),
                    _to_dataset([e.neg_segment_ids for e in examples]),
                    _to_dataset([e.neg_attention_mask for e in examples]),
                )
            )
        elif with_pos_sequence:
            dataset = tf.data.Dataset.zip(
                (
                    _to_dataset([e.input_ids for e in examples]),
                    _to_dataset([e.segment_ids for e in examples]),
                    _to_dataset([e.attention_mask for e in examples]),
                    _to_dataset([e.pos_input_ids for e in examples]),
                    _to_dataset([e.pos_segment_ids for e in examples]),
                    _to_dataset([e.pos_attention_mask for e in examples]),
                )
            )
        else:
            dataset = tf.data.Dataset.zip(
                (
                    _to_dataset([e.input_ids for e in examples]),
                    _to_dataset([e.segment_ids for e in examples]),
                    _to_dataset([e.attention_mask for e in examples]),
                )
            )
        return dataset

    @classmethod
    def _filter_dataset(
        cls, dataset, max_sequence_length=512, with_pos_sequence=False, with_neg_sequence=False, **kwargs
    ):
        if with_neg_sequence:
            dataset = dataset.filter(
                lambda a, b, c, e, f, g, h, i, j: tf.logical_and(
                    tf.size(a) <= max_sequence_length,
                    tf.logical_and(tf.size(e) <= max_sequence_length, tf.size(h) <= max_sequence_length),
                )
            )
        elif with_pos_sequence:
            dataset = dataset.filter(
                lambda a, b, c, e, f, g: tf.logical_and(
                    tf.size(a) <= max_sequence_length, tf.size(e) <= max_sequence_length
                )
            )
        else:
            dataset = dataset.filter(lambda a, b, c: tf.size(a) <= max_sequence_length)
        return dataset

    @classmethod
    def _bucketing(
        cls,
        dataset,
        with_pos_sequence=False,
        with_neg_sequence=False,
        batch_size=64,
        pad_id=0,
        bucket_boundaries=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        bucket_batch_sizes=None,
        drop_remainder=True,
        **kwargs
    ):
        if bucket_batch_sizes is None:
            bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
        assert (
            len(bucket_batch_sizes) == len(bucket_boundaries) + 1
        ), "len(bucket_batch_sizes) should equals len(bucket_doundaries) + 1"

        pad = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        if with_neg_sequence:
            dataset = dataset.apply(
                tf.data.experimental.bucket_by_sequence_length(
                    element_length_func=lambda a, b, c, e, f, g, h, i, j: tf.maximum(
                        tf.size(a), tf.maximum(tf.size(e), tf.size(h))
                    ),
                    bucket_boundaries=bucket_boundaries,
                    bucket_batch_sizes=bucket_batch_sizes,
                    padded_shapes=([None,], [None,], [None,], [None,], [None,], [None,], [None,], [None,], [None,]),
                    padding_values=(pad, pad, pad, pad, pad, pad, pad, pad, pad),
                    drop_remainder=drop_remainder,
                )
            ).prefetch(tf.data.AUTOTUNE)
        elif with_pos_sequence:
            dataset = dataset.apply(
                tf.data.experimental.bucket_by_sequence_length(
                    element_length_func=lambda a, b, c, e, f, g: tf.maximum(tf.size(a), tf.size(e)),
                    bucket_boundaries=bucket_boundaries,
                    bucket_batch_sizes=bucket_batch_sizes,
                    padded_shapes=([None,], [None,], [None,], [None,], [None,], [None,]),
                    padding_values=(pad, pad, pad, pad, pad, pad),
                    drop_remainder=drop_remainder,
                )
            ).prefetch(tf.data.AUTOTUNE)
        else:
            dataset = dataset.apply(
                tf.data.experimental.bucket_by_sequence_length(
                    element_length_func=lambda a, b, c: tf.size(a),
                    bucket_boundaries=bucket_boundaries,
                    bucket_batch_sizes=bucket_batch_sizes,
                    padded_shapes=([None, ], [None, ], [None, ]),
                    padding_values=(pad, pad, pad),
                    drop_remainder=drop_remainder,
                )
            ).prefetch(tf.data.AUTOTUNE)
        # fmt: on
        return dataset

    @classmethod
    def _to_dict(cls, dataset, with_pos_sequence=False, with_neg_sequence=False, **kwargs):
        if with_neg_sequence:
            dataset = dataset.map(
                lambda a, b, c, e, f, g, h, i, j: (
                    {
                        "input_ids": a,
                        "segment_ids": b,
                        "attention_mask": c,
                        "pos_input_ids": e,
                        "pos_segment_ids": f,
                        "pos_attention_mask": g,
                        "neg_input_ids": h,
                        "neg_segment_ids": i,
                        "neg_attention_mask": j,
                    },
                    None,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            ).prefetch(tf.data.AUTOTUNE)
            return dataset
        elif with_pos_sequence:
            dataset = dataset.map(
                lambda a, b, c, e, f, g: (
                    {
                        "input_ids": a,
                        "segment_ids": b,
                        "attention_mask": c,
                        "pos_input_ids": e,
                        "pos_segment_ids": f,
                        "pos_attention_mask": g,
                    },
                    None,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            ).prefetch(tf.data.AUTOTUNE)
        else:
            dataset = dataset.map(
                lambda x, y, z: ({"input_ids": x, "segment_ids": y, "attention_mask": z}, None),
                num_parallel_calls=tf.data.AUTOTUNE,
            ).prefetch(tf.data.AUTOTUNE)
        return dataset

    @classmethod
    def _auto_shard(cls, dataset, auto_shard_policy=None, **kwargs):
        if auto_shard_policy is not None:
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = auto_shard_policy
            dataset = dataset.with_options(options)
        return dataset

    @classmethod
    def _read_jsonl_examples(
        cls, input_files, fn, with_pos_sequence=False, with_neg_sequence=False, **kwargs
    ) -> List[SimCSEExample]:
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
                    examples = fn(
                        instance, with_pos_sequence=with_pos_sequence, with_neg_sequence=with_neg_sequence, **kwargs
                    )
                    all_examples.extend(examples)
        return all_examples

    @classmethod
    def _read_tfrecord(cls, input_files, with_pos_sequence=False, with_neg_sequence=False, **kwargs):
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
        }
        if with_pos_sequence:
            features.update(
                {
                    "pos_input_ids": tf.io.VarLenFeature(tf.int64),
                    "pos_segment_ids": tf.io.VarLenFeature(tf.int64),
                    "pos_attention_mask": tf.io.VarLenFeature(tf.int64),
                }
            )
        if with_neg_sequence:
            features.update(
                {
                    "neg_input_ids": tf.io.VarLenFeature(tf.int64),
                    "neg_segment_ids": tf.io.VarLenFeature(tf.int64),
                    "neg_attention_mask": tf.io.VarLenFeature(tf.int64),
                }
            )
        dataset = dataset.map(
            lambda x: tf.io.parse_example(x, features),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)

        if with_neg_sequence:
            dataset = dataset.map(
                lambda x: (
                    tf.cast(tf.sparse.to_dense(x["input_ids"]), tf.int32),
                    tf.cast(tf.sparse.to_dense(x["segment_ids"]), tf.int32),
                    tf.cast(tf.sparse.to_dense(x["attention_mask"]), tf.int32),
                    tf.cast(tf.sparse.to_dense(x["pos_input_ids"]), tf.int32),
                    tf.cast(tf.sparse.to_dense(x["pos_segment_ids"]), tf.int32),
                    tf.cast(tf.sparse.to_dense(x["pos_attention_mask"]), tf.int32),
                    tf.cast(tf.sparse.to_dense(x["neg_input_ids"]), tf.int32),
                    tf.cast(tf.sparse.to_dense(x["neg_segment_ids"]), tf.int32),
                    tf.cast(tf.sparse.to_dense(x["neg_attention_mask"]), tf.int32),
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            ).prefetch(tf.data.AUTOTUNE)
        elif with_pos_sequence:
            dataset = dataset.map(
                lambda x: (
                    tf.cast(tf.sparse.to_dense(x["input_ids"]), tf.int32),
                    tf.cast(tf.sparse.to_dense(x["segment_ids"]), tf.int32),
                    tf.cast(tf.sparse.to_dense(x["attention_mask"]), tf.int32),
                    tf.cast(tf.sparse.to_dense(x["pos_input_ids"]), tf.int32),
                    tf.cast(tf.sparse.to_dense(x["pos_segment_ids"]), tf.int32),
                    tf.cast(tf.sparse.to_dense(x["pos_attention_mask"]), tf.int32),
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            ).prefetch(tf.data.AUTOTUNE)
        else:
            dataset = dataset.map(
                lambda x: (
                    tf.cast(tf.sparse.to_dense(x["input_ids"]), tf.int32),
                    tf.cast(tf.sparse.to_dense(x["segment_ids"]), tf.int32),
                    tf.cast(tf.sparse.to_dense(x["attention_mask"]), tf.int32),
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            ).prefetch(tf.data.AUTOTUNE)
        return dataset
