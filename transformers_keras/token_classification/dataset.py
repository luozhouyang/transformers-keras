import logging
from collections import namedtuple
from typing import List

import tensorflow as tf

TokenClassificationExample = namedtuple(
    "TokenClassificationExample", ["tokens", "labels", "input_ids", "segment_ids", "attention_mask", "label_ids"]
)


class _TokenClassificationDatasetTransform:
    """Dataset transformation for token classification."""

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

    @classmethod
    def _example_to_tfrecord(cls, example: TokenClassificationExample, **kwargs):
        feature = {
            "input_ids": cls._int64_feature([int(x) for x in example.input_ids]),
            "segment_ids": cls._int64_feature([int(x) for x in example.segment_ids]),
            "attention_mask": cls._int64_feature([int(x) for x in example.attention_mask]),
            "label_ids": cls._int64_feature([int(x) for x in example.label_ids]),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    @classmethod
    def _int64_feature(cls, values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


class TokenClassificationDataset:
    """Dataset for token classification."""

    @classmethod
    def examples_to_tfrecord(cls, examples, output_file, **kwargs):
        _TokenClassificationDatasetTransform.examples_to_tfrecord(examples, output_file, **kwargs)
        logging.info("Done!")

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
        auto_shard_policy=None,
        drop_remainder=False,
        **kwargs
    ):
        dataset = cls._read_tfrecord(input_files, **kwargs)
        dataset = dataset.filter(lambda a, b, c, y: tf.size(a) <= max_sequence_length)
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
            **kwargs
        )
        dataset = cls._to_dict(dataset)
        dataset = cls._auto_shard(dataset, auto_shard_policy=auto_shard_policy)
        return dataset

    @classmethod
    def from_examples(
        cls,
        examples: List[TokenClassificationExample],
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
        logging.info("Number of examples: %d", len(examples))
        cls._show_examples(examples, n=5, verbose=verbose, **kwargs)
        dataset = cls._zip_dataset(examples)
        dataset = dataset.filter(lambda a, b, c, y: tf.size(a) <= max_sequence_length)
        if repeat is not None:
            dataset = dataset.repeat(repeat)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)
        dataset = cls._bucketing(
            dataset,
            batch_size=batch_size,
            pad_id=pad_id,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            dorp_remainder=drop_remainder,
            **kwargs
        )
        dataset = cls._to_dict(dataset)
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
    def _zip_dataset(cls, examples, **kwargs):
        """zip dataset"""

        def _to_dataset(x, dtype=tf.int32):
            x = tf.ragged.constant(x, dtype=dtype)
            d = tf.data.Dataset.from_tensor_slices(x)
            d = d.map(lambda x: x)
            return d

        dataset = tf.data.Dataset.zip(
            (
                _to_dataset([e.input_ids for e in examples]),
                _to_dataset([e.segment_ids for e in examples]),
                _to_dataset([e.attention_mask for e in examples]),
                _to_dataset([e.label_ids for e in examples]),
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
        ), "len(bucket_batch_size) should equals len(bucket_doundaries) + 1"

        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
            element_length_func=lambda a, b, c, y: tf.size(a),
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            padded_shapes=([None, ], [None, ], [None, ], [None, ]),
            padding_values=(pad_id, pad_id, pad_id, pad_id),
            drop_remainder=drop_remainder,
        )).prefetch(tf.data.AUTOTUNE)
        # fmt: on
        return dataset

    @classmethod
    def _to_dict(cls, dataset):
        dataset = dataset.map(
            lambda a, b, c, y: ({"input_ids": a, "segment_ids": b, "attention_mask": c}, y),
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
            "label_ids": tf.io.VarLenFeature(tf.int64),
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
                tf.cast(tf.sparse.to_dense(x["label_ids"]), tf.int32),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)
        return dataset
