from collections import namedtuple
from typing import List

import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
from transformers_keras.dataset_utils import AbstractDataset

UnsupervisedSimCSEExample = namedtuple(
    "UnsupervisedSimCSEExample",
    [
        "tokens",
        "input_ids",
        "segment_ids",
        "attention_mask",
    ],
)
SupervisedSimCSEExample = namedtuple(
    "SupervisedSimCSEExample",
    [
        "tokens",
        "input_ids",
        "segment_ids",
        "attention_mask",
        "pos_tokens",
        "pos_input_ids",
        "pos_segment_ids",
        "pos_attention_mask",
    ],
)

HardNegativeSimCSEExample = namedtuple(
    "HardNegativeSimCSEExample",
    [
        "tokens",
        "input_ids",
        "segment_ids",
        "attention_mask",
        "pos_tokens",
        "pos_input_ids",
        "pos_segment_ids",
        "pos_attention_mask",
        "neg_tokens",
        "neg_input_ids",
        "neg_segment_ids",
        "neg_attention_mask",
    ],
)


class UnsupervisedSimCSEDataset(AbstractDataset):
    """Dataset builder for unsupervised SimCSE"""

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
        drop_remainder=True,
        **kwargs
    ):
        dataset = dataset.filter(lambda a, b, c: tf.size(a) <= max_sequence_length)
        if repeat is not None:
            dataset = dataset.repeat(repeat)
        dataset = dataset.shuffle(
            buffer_size=buffer_size,
            seed=seed,
            reshuffle_each_iteration=reshuffle_each_iteration,
        )
        pad = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        dataset = cls._bucketing(
            dataset,
            element_length_func=lambda a, b, c: tf.size(a),
            padded_shapes=([None, ], [None, ], [None, ]),
            padding_values=(pad, pad, pad),
            batch_size=batch_size,
            pad_id=pad_id,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            drop_remainder=drop_remainder,
            **kwargs,
        )
        # fmt: on
        dataset = cls._to_dict(dataset, **kwargs)
        dataset = cls._auto_shard(dataset, auto_shard_policy=auto_shard_policy)
        return dataset

    @classmethod
    def _zip_dataset(cls, examples: List[UnsupervisedSimCSEExample], **kwargs):
        def _to_dataset(x, dtype=tf.int32):
            x = tf.ragged.constant(x, dtype=dtype)
            d = tf.data.Dataset.from_tensor_slices(x)
            d = d.map(lambda x: x)
            return d

        dataset = tf.data.Dataset.zip(
            (
                _to_dataset([e.input_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.segment_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.attention_mask for e in examples], dtype=tf.int32),
            )
        )
        return dataset

    @classmethod
    def _to_dict(cls, dataset, **kwargs):
        dataset = dataset.map(
            lambda x, y, z: ({"input_ids": x, "segment_ids": y, "attention_mask": z}, None),
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        return dataset

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
            sequence = instance[kwargs.get("sequence_key", "sequence")]
            seq_encoding = tokenizer.encode(sequence)
            example = UnsupervisedSimCSEExample(
                tokens=seq_encoding.tokens,
                input_ids=seq_encoding.ids,
                segment_ids=seq_encoding.type_ids,
                attention_mask=seq_encoding.attention_mask,
            )
            examples.append(example)
        return examples

    @classmethod
    def _parse_tfrecord(cls, dataset, **kwargs):
        features = {
            "input_ids": tf.io.VarLenFeature(tf.int64),
            "segment_ids": tf.io.VarLenFeature(tf.int64),
            "attention_mask": tf.io.VarLenFeature(tf.int64),
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
            ),
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        return dataset

    @classmethod
    def _example_to_tfrecord(cls, example, **kwargs):
        feature = {
            "input_ids": cls._int64_feature([int(x) for x in example.input_ids]),
            "segment_ids": cls._int64_feature([int(x) for x in example.segment_ids]),
            "attention_mask": cls._int64_feature([int(x) for x in example.attention_mask]),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))


class SupervisedSimCSEDataset(AbstractDataset):
    """Dataset builder for supervised SimCSE"""

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
        drop_remainder=True,
        **kwargs
    ):
        dataset = dataset.filter(
            lambda a, b, c, e, f, g: tf.logical_and(tf.size(a) <= max_sequence_length, tf.size(e) <= max_sequence_length)
        )
        if repeat is not None:
            dataset = dataset.repeat(repeat)
        dataset = dataset.shuffle(
            buffer_size=buffer_size,
            seed=seed,
            reshuffle_each_iteration=reshuffle_each_iteration,
        )
        pad = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        dataset = cls._bucketing(
            dataset,
            element_length_func=lambda a, b, c, e, f, g: tf.maximum(tf.size(a), tf.size(e)),
            padded_shapes=([None,], [None,], [None,], [None,], [None,], [None,]),
            padding_values=(pad, pad, pad, pad, pad, pad),
            batch_size=batch_size,
            pad_id=pad_id,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            drop_remainder=drop_remainder,
            **kwargs,
        )
        # fmt: on
        dataset = cls._to_dict(dataset, **kwargs)
        dataset = cls._auto_shard(dataset, auto_shard_policy=auto_shard_policy)
        return dataset

    @classmethod
    def _zip_dataset(cls, examples: List[SupervisedSimCSEExample], **kwargs):
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
                _to_dataset([e.pos_input_ids for e in examples]),
                _to_dataset([e.pos_segment_ids for e in examples]),
                _to_dataset([e.pos_attention_mask for e in examples]),
            )
        )
        return dataset

    @classmethod
    def _to_dict(cls, dataset, **kwargs):
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
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        return dataset

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
            sequence = instance[kwargs.get("sequence_key", "sequence")]
            seq_encoding = tokenizer.encode(sequence)
            pos_sequence = instance[kwargs.get("pos_sequence_key", "pos_sequence")]
            pos_encoding = tokenizer.encode(pos_sequence)
            example = SupervisedSimCSEExample(
                tokens=seq_encoding.tokens,
                input_ids=seq_encoding.ids,
                segment_ids=seq_encoding.type_ids,
                attention_mask=seq_encoding.attention_mask,
                pos_tokens=pos_encoding.tokens,
                pos_input_ids=pos_encoding.ids,
                pos_segment_ids=pos_encoding.type_ids,
                pos_attention_mask=pos_encoding.attention_mask,
            )
            examples.append(example)
        return examples

    @classmethod
    def _parse_tfrecord(cls, dataset, **kwargs):
        features = {
            "input_ids": tf.io.VarLenFeature(tf.int64),
            "segment_ids": tf.io.VarLenFeature(tf.int64),
            "attention_mask": tf.io.VarLenFeature(tf.int64),
            "pos_input_ids": tf.io.VarLenFeature(tf.int64),
            "pos_segment_ids": tf.io.VarLenFeature(tf.int64),
            "pos_attention_mask": tf.io.VarLenFeature(tf.int64),
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
                tf.cast(tf.sparse.to_dense(x["pos_input_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["pos_segment_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["pos_attention_mask"]), tf.int32),
            ),
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        return dataset

    @classmethod
    def _example_to_tfrecord(cls, example, **kwargs):
        feature = {
            "input_ids": cls._int64_feature([int(x) for x in example.input_ids]),
            "segment_ids": cls._int64_feature([int(x) for x in example.segment_ids]),
            "attention_mask": cls._int64_feature([int(x) for x in example.attention_mask]),
            "pos_input_ids": cls._int64_feature([int(x) for x in example.pos_input_ids]),
            "pos_segment_ids": cls._int64_feature([int(x) for x in example.pos_segment_ids]),
            "pos_attention_mask": cls._int64_feature([int(x) for x in example.pos_attention_mask]),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))


class HardNegativeSimCSEDataset(AbstractDataset):
    """Dataset builder for hard negavtive SimCSE"""

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
        drop_remainder=True,
        **kwargs
    ):
        dataset = dataset.filter(
            lambda a, b, c, e, f, g, h, i, j: tf.logical_and(
                tf.size(a) <= max_sequence_length,
                tf.logical_and(tf.size(e) <= max_sequence_length, tf.size(h) <= max_sequence_length),
            )
        )
        if repeat is not None:
            dataset = dataset.repeat(repeat)
        dataset = dataset.shuffle(
            buffer_size=buffer_size,
            seed=seed,
            reshuffle_each_iteration=reshuffle_each_iteration,
        )
        pad = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        dataset = cls._bucketing(
            dataset,
            element_length_func=lambda a, b, c, e, f, g, h, i, j: tf.maximum(
                tf.size(a), tf.maximum(tf.size(e), tf.size(h))
            ),
            padded_shapes=([None,], [None,], [None,], [None,], [None,], [None,], [None,], [None,], [None,]),
            padding_values=(pad, pad, pad, pad, pad, pad, pad, pad, pad),
            batch_size=batch_size,
            pad_id=pad_id,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            drop_remainder=drop_remainder,
            **kwargs,
        )
        # fmt: on
        dataset = cls._to_dict(dataset, **kwargs)
        dataset = cls._auto_shard(dataset, auto_shard_policy=auto_shard_policy)
        return dataset

    @classmethod
    def _zip_dataset(cls, examples: List[UnsupervisedSimCSEExample], **kwargs):
        def _to_dataset(x, dtype=tf.int32):
            x = tf.ragged.constant(x, dtype=dtype)
            d = tf.data.Dataset.from_tensor_slices(x)
            d = d.map(lambda x: x)
            return d

        dataset = tf.data.Dataset.zip(
            (
                _to_dataset([e.input_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.segment_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.attention_mask for e in examples], dtype=tf.int32),
                _to_dataset([e.pos_input_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.pos_segment_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.pos_attention_mask for e in examples], dtype=tf.int32),
                _to_dataset([e.neg_input_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.neg_segment_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.neg_attention_mask for e in examples], dtype=tf.int32),
            )
        )
        return dataset

    @classmethod
    def _to_dict(cls, dataset, **kwargs):
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
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        return dataset

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
            sequence = instance[kwargs.get("sequence_key", "sequence")]
            seq_encoding = tokenizer.encode(sequence)
            pos_sequence = instance[kwargs.get("pos_sequence_key", "pos_sequence")]
            pos_encoding = tokenizer.encode(pos_sequence)
            neg_sequence = instance[kwargs.get("neg_sequence_key", "neg_sequence")]
            neg_encoding = tokenizer.encode(neg_sequence)
            example = HardNegativeSimCSEExample(
                tokens=seq_encoding.tokens,
                input_ids=seq_encoding.ids,
                segment_ids=seq_encoding.type_ids,
                attention_mask=seq_encoding.attention_mask,
                pos_tokens=pos_encoding.tokens,
                pos_input_ids=pos_encoding.ids,
                pos_segment_ids=pos_encoding.type_ids,
                pos_attention_mask=pos_encoding.attention_mask,
                neg_tokens=neg_encoding.tokens,
                neg_input_ids=neg_encoding.ids,
                neg_segment_ids=neg_encoding.type_ids,
                neg_attention_mask=neg_encoding.attention_mask,
            )
            examples.append(example)
        return examples

    @classmethod
    def _parse_tfrecord(cls, dataset, **kwargs):
        features = {
            "input_ids": tf.io.VarLenFeature(tf.int64),
            "segment_ids": tf.io.VarLenFeature(tf.int64),
            "attention_mask": tf.io.VarLenFeature(tf.int64),
            "neg_input_ids": tf.io.VarLenFeature(tf.int64),
            "neg_segment_ids": tf.io.VarLenFeature(tf.int64),
            "neg_attention_mask": tf.io.VarLenFeature(tf.int64),
            "pos_input_ids": tf.io.VarLenFeature(tf.int64),
            "pos_segment_ids": tf.io.VarLenFeature(tf.int64),
            "pos_attention_mask": tf.io.VarLenFeature(tf.int64),
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
                tf.cast(tf.sparse.to_dense(x["pos_input_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["pos_segment_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["pos_attention_mask"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["neg_input_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["neg_segment_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["neg_attention_mask"]), tf.int32),
            ),
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        return dataset

    @classmethod
    def _example_to_tfrecord(cls, example, **kwargs):
        feature = {
            "input_ids": cls._int64_feature([int(x) for x in example.input_ids]),
            "segment_ids": cls._int64_feature([int(x) for x in example.segment_ids]),
            "attention_mask": cls._int64_feature([int(x) for x in example.attention_mask]),
            "pos_input_ids": cls._int64_feature([int(x) for x in example.pos_input_ids]),
            "pos_segment_ids": cls._int64_feature([int(x) for x in example.pos_segment_ids]),
            "pos_attention_mask": cls._int64_feature([int(x) for x in example.pos_attention_mask]),
            "neg_input_ids": cls._int64_feature([int(x) for x in example.neg_input_ids]),
            "neg_segment_ids": cls._int64_feature([int(x) for x in example.neg_segment_ids]),
            "neg_attention_mask": cls._int64_feature([int(x) for x in example.neg_attention_mask]),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))
