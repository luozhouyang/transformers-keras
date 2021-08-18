import json
import logging
import os
import re
from collections import namedtuple
from typing import List

import tensorflow as tf
from tokenizers import BertWordPieceTokenizer

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
    def from_jsonl_files(cls, input_file, vocab_file, **kwargs):
        examples = cls.jsonl_to_examples(input_file, vocab_file, **kwargs)
        return cls.from_examples(examples, **kwargs)

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
    def _read_tfrecord(cls, input_files, with_pos_sequence=False, with_neg_sequence=False, **kwargs):
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
        }
        if with_neg_sequence:
            features.update(
                {
                    "neg_input_ids": tf.io.VarLenFeature(tf.int64),
                    "neg_segment_ids": tf.io.VarLenFeature(tf.int64),
                    "neg_attention_mask": tf.io.VarLenFeature(tf.int64),
                    "pos_input_ids": tf.io.VarLenFeature(tf.int64),
                    "pos_segment_ids": tf.io.VarLenFeature(tf.int64),
                    "pos_attention_mask": tf.io.VarLenFeature(tf.int64),
                }
            )
        elif with_pos_sequence:
            features.update(
                {
                    "pos_input_ids": tf.io.VarLenFeature(tf.int64),
                    "pos_segment_ids": tf.io.VarLenFeature(tf.int64),
                    "pos_attention_mask": tf.io.VarLenFeature(tf.int64),
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

    @classmethod
    def jsonl_to_tfrecord(
        cls, input_files, vocab_file, output_files, with_pos_sequence=False, with_neg_sequence=False, **kwargs
    ):
        examples = cls.jsonl_to_examples(
            input_files, vocab_file, with_pos_sequence=with_pos_sequence, with_neg_sequence=with_neg_sequence, **kwargs
        )
        cls.examples_to_tfrecord(
            examples, output_files, with_pos_sequence=with_pos_sequence, with_neg_sequence=with_neg_sequence, **kwargs
        )
        logging.info("Convert data from jsonl to tfrecord finished.")

    @classmethod
    def jsonl_to_examples(cls, input_files, vocab_file, with_pos_sequence=False, with_neg_sequence=False, **kwargs):
        if isinstance(input_files, str):
            input_files = [input_files]
        examples = []
        tokenizer = BertWordPieceTokenizer.from_file(vocab_file, lowercase=kwargs.get("do_lower_case", True))
        for f in input_files:
            with open(f, mode="rt", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    instance = json.loads(line)
                    example = cls._instance_to_example(
                        instance,
                        tokenizer,
                        with_pos_sequence=with_pos_sequence,
                        with_neg_sequence=with_neg_sequence,
                        **kwargs,
                    )
                    examples.append(example)
        logging.info("Collected %d examples in total.", len(examples))
        return examples

    @classmethod
    def _instance_to_example(
        cls, instance, tokenizer: BertWordPieceTokenizer, with_pos_sequence=False, with_neg_sequence=False, **kwargs
    ):
        sequence = instance[kwargs.get("sequence_key", "sequence")]
        seq_encoding = tokenizer.encode(sequence)
        if with_neg_sequence:
            pos_sequence = instance[kwargs.get("pos_sequence_key", "pos_sequence")]
            pos_encoding = tokenizer.encode(pos_sequence)
            neg_sequence = instance[kwargs.get("neg_sequence_key", "neg_sequence")]
            neg_encoding = tokenizer.encode(neg_sequence)
            example = SimCSEExample(
                sequence=sequence,
                input_ids=seq_encoding.ids,
                segment_ids=seq_encoding.type_ids,
                attention_mask=seq_encoding.attention_mask,
                pos_sequence=pos_sequence,
                pos_input_ids=pos_encoding.ids,
                pos_segment_ids=pos_encoding.type_ids,
                pos_attention_mask=pos_encoding.attention_mask,
                neg_sequence=neg_sequence,
                neg_input_ids=neg_encoding.ids,
                neg_segment_ids=neg_encoding.type_ids,
                neg_attention_mask=neg_encoding.attention_mask,
            )
            return example
        if with_pos_sequence:
            pos_sequence = instance[kwargs.get("pos_sequence_key", "pos_sequence")]
            pos_encoding = tokenizer.encode(pos_sequence)
            example = SimCSEExample(
                sequence=sequence,
                input_ids=seq_encoding.ids,
                segment_ids=seq_encoding.type_ids,
                attention_mask=seq_encoding.attention_mask,
                pos_sequence=pos_sequence,
                pos_input_ids=pos_encoding.ids,
                pos_segment_ids=pos_encoding.type_ids,
                pos_attention_mask=pos_encoding.attention_mask,
                neg_sequence=None,
                neg_input_ids=None,
                neg_segment_ids=None,
                neg_attention_mask=None,
            )
            return example
        example = SimCSEExample(
            sequence=sequence,
            input_ids=seq_encoding.ids,
            segment_ids=seq_encoding.type_ids,
            attention_mask=seq_encoding.attention_mask,
            pos_sequence=None,
            pos_input_ids=None,
            pos_segment_ids=None,
            pos_attention_mask=None,
            neg_sequence=None,
            neg_input_ids=None,
            neg_segment_ids=None,
            neg_attention_mask=None,
        )
        return example

    @classmethod
    def examples_to_tfrecord(cls, examples, output_files, with_pos_sequence=False, with_neg_sequence=False, **kwargs):
        if isinstance(output_files, str):
            output_files = [output_files]
        writers = [tf.io.TFRecordWriter(f) for f in output_files]
        idx = 0
        for example in examples:
            tfrecord_example = cls._example_to_tfrecord(
                example, with_pos_sequence=with_pos_sequence, with_neg_sequence=with_neg_sequence, **kwargs
            )
            writers[idx].write(tfrecord_example.SerializeToString())
            idx += 1
            idx = idx % len(writers)
        for w in writers:
            w.close()

    @classmethod
    def _example_to_tfrecord(cls, example: SimCSEExample, with_pos_sequence=False, with_neg_sequence=False, **kwargs):
        feature = {
            "input_ids": cls._int64_feature([int(x) for x in example.input_ids]),
            "segment_ids": cls._int64_feature([int(x) for x in example.segment_ids]),
            "attention_mask": cls._int64_feature([int(x) for x in example.attention_mask]),
        }
        if with_neg_sequence:
            feature.update(
                {
                    "pos_input_ids": cls._int64_feature([int(x) for x in example.pos_input_ids]),
                    "pos_segment_ids": cls._int64_feature([int(x) for x in example.pos_segment_ids]),
                    "pos_attention_mask": cls._int64_feature([int(x) for x in example.pos_attention_mask]),
                    "neg_input_ids": cls._int64_feature([int(x) for x in example.neg_input_ids]),
                    "neg_segment_ids": cls._int64_feature([int(x) for x in example.neg_segment_ids]),
                    "neg_attention_mask": cls._int64_feature([int(x) for x in example.neg_attention_mask]),
                }
            )
        elif with_pos_sequence:
            feature.update(
                {
                    "pos_input_ids": cls._int64_feature([int(x) for x in example.pos_input_ids]),
                    "pos_segment_ids": cls._int64_feature([int(x) for x in example.pos_segment_ids]),
                    "pos_attention_mask": cls._int64_feature([int(x) for x in example.pos_attention_mask]),
                }
            )
        return tf.train.Example(features=tf.train.Features(feature=feature))

    @classmethod
    def _int64_feature(cls, values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
