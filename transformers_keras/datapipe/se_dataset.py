"""Dataset builder for sentence embeddings."""
from collections import namedtuple
from typing import List

import tensorflow as tf
from tokenizers import BertWordPieceTokenizer

from .abc_dataset import AbstractDataPipe, DatasetForBert

ExampleForUnsupervisedSimCSE = namedtuple(
    "ExampleForUnsupervisedSimCSE",
    [
        "tokens",
        "input_ids",
        "segment_ids",
        "attention_mask",
    ],
)
ExampleForSupervisedSimCSE = namedtuple(
    "ExampleForSupervisedSimCSE",
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

ExampleForHardNegativeSimCSE = namedtuple(
    "ExampleForHardNegativeSimCSE",
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


class DatasetForUnupervisedSimCSE(DatasetForBert):
    """Dataset for unsup SimCSE"""

    def __init__(self, input_file, vocab_file, add_special_tokens=True, sequence_key="sequence", **kwargs) -> None:
        super().__init__(vocab_file, **kwargs)
        self.instances = self._read_jsonl_files(input_file, **kwargs)
        self.add_special_tokens = add_special_tokens
        self.sequence_key = sequence_key

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> ExampleForUnsupervisedSimCSE:
        instance = self.instances[index]
        sequence = instance[self.sequence_key]
        seq_encoding = self.tokenizer.encode(sequence, add_special_tokens=self.add_special_tokens)
        example = ExampleForUnsupervisedSimCSE(
            tokens=seq_encoding.tokens,
            input_ids=seq_encoding.ids,
            segment_ids=seq_encoding.type_ids,
            attention_mask=seq_encoding.attention_mask,
        )
        return example


class DatasetForSupervisedSimCSE(DatasetForUnupervisedSimCSE):
    """Dataset for supervised SimCSE"""

    def __init__(self, input_file, vocab_file, pos_sequence_key="pos_sequence", **kwargs) -> None:
        super().__init__(input_file, vocab_file, **kwargs)
        self.pos_sequence_key = pos_sequence_key

    def __getitem__(self, index) -> ExampleForSupervisedSimCSE:
        instance = self.instances[index]
        seq, pos_seq = instance[self.sequence_key], instance[self.pos_sequence_key]
        seq_encoding = self.tokenizer.encode(seq, add_special_tokens=self.add_special_tokens)
        pos_encoding = self.tokenizer.encode(pos_seq, add_special_tokens=self.add_special_tokens)
        example = ExampleForSupervisedSimCSE(
            tokens=seq_encoding.tokens,
            input_ids=seq_encoding.ids,
            segment_ids=seq_encoding.type_ids,
            attention_mask=seq_encoding.attention_mask,
            pos_tokens=pos_encoding.tokens,
            pos_input_ids=pos_encoding.ids,
            pos_segment_ids=pos_encoding.type_ids,
            pos_attention_mask=pos_encoding.attention_mask,
        )
        return example


class DatasetForHardNegativeSimCSE(DatasetForSupervisedSimCSE):
    """Dataset for hard negative SimCSE"""

    def __init__(self, input_file, vocab_file, neg_sequence_key="neg_sequence", **kwargs) -> None:
        super().__init__(input_file, vocab_file, **kwargs)
        self.neg_sequence_key = neg_sequence_key

    def __getitem__(self, index) -> ExampleForHardNegativeSimCSE:
        instance = self.instances[index]
        pos_seq, neg_seq = instance[self.pos_sequence_key], instance[self.neg_sequence_key]
        seq_encoding = self.tokenizer.encode(instance[self.sequence_key], add_special_tokens=self.add_special_tokens)
        pos_encoding = self.tokenizer.encode(pos_seq, add_special_tokens=self.add_special_tokens)
        neg_encoding = self.tokenizer.encode(neg_seq, add_special_tokens=self.add_special_tokens)
        example = ExampleForHardNegativeSimCSE(
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
        return example


class DataPipeForUnsupervisedSimCSE(AbstractDataPipe):
    """Dataset builder for unsupervised SimCSE"""

    @classmethod
    def _dataset_from_jsonl_files(cls, input_files, vocab_file, **kwargs) -> DatasetForUnupervisedSimCSE:
        return DatasetForUnupervisedSimCSE(input_files, vocab_file, **kwargs)

    @classmethod
    def _transform_examples_to_dataset(cls, examples, **kwargs) -> tf.data.Dataset:
        """Zip examples to dataset"""

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
    def _transform_example_to_tfrecord(cls, example, **kwargs) -> tf.train.Example:
        feature = {
            "input_ids": cls._int64_feature([int(x) for x in example.input_ids]),
            "segment_ids": cls._int64_feature([int(x) for x in example.segment_ids]),
            "attention_mask": cls._int64_feature([int(x) for x in example.attention_mask]),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    @classmethod
    def _parse_tfrecord_dataset(cls, dataset: tf.data.Dataset, **kwargs) -> tf.data.Dataset:
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
    def _filter(cls, dataset, max_sequence_length=512, **kwargs):
        dataset = dataset.filter(lambda a, b, c: tf.size(a) <= max_sequence_length)
        return dataset

    @classmethod
    def _bucket_padding(
        cls,
        dataset,
        batch_size=32,
        pad_id=0,
        bucket_boundaries=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        bucket_batch_sizes=None,
        drop_remainder=False,
        **kwargs
    ):
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        dataset = cls._bucketing(
            dataset,
            element_length_func=lambda a, b, c: tf.size(a),
            padded_shapes=([None, ], [None, ], [None, ]),
            padding_values=(pad_id, pad_id, pad_id),
            batch_size=batch_size,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            drop_remainder=drop_remainder,
            **kwargs,
        )
        # fmt: on
        return dataset

    @classmethod
    def _batch_padding(cls, dataset, batch_size=32, pad_id=0, drop_remainder=False, **kwargs):
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([None, ], [None, ], [None, ]),
            padding_values=(pad_id, pad_id, pad_id),
            drop_remainder=drop_remainder,
        )
        # fmt: on
        return dataset

    @classmethod
    def _fixed_padding(cls, dataset, batch_size=32, pad_id=0, max_sequence_length=512, drop_remainder=False, **kwargs):
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        maxlen = tf.constant(max_sequence_length, dtype=tf.int32)
        # fmt: off
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([None, ], [None, ], [None, ]),
            padding_values=(pad_id, pad_id, pad_id),
            drop_remainder=drop_remainder,
        )
        # fmt: on
        return dataset

    @classmethod
    def _to_dict(cls, dataset, to_dict=True, **kwargs):
        if not to_dict:
            return dataset
        dataset = dataset.map(
            lambda x, y, z: ({"input_ids": x, "segment_ids": y, "attention_mask": z}, None),
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        return dataset


class DataPipeForSupervisedSimCSE(AbstractDataPipe):
    """Dataset builder for supervised SimCSE"""

    @classmethod
    def _dataset_from_jsonl_files(cls, input_files, vocab_file, **kwargs) -> DatasetForSupervisedSimCSE:
        return DatasetForSupervisedSimCSE(input_files, vocab_file, **kwargs)

    @classmethod
    def _transform_examples_to_dataset(cls, examples, **kwargs) -> tf.data.Dataset:
        """Zip examples to dataset"""

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
    def _parse_tfrecord_dataset(cls, dataset: tf.data.Dataset, **kwargs) -> tf.data.Dataset:
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
    def _transform_example_to_tfrecord(cls, example, **kwargs) -> tf.train.Example:
        feature = {
            "input_ids": cls._int64_feature([int(x) for x in example.input_ids]),
            "segment_ids": cls._int64_feature([int(x) for x in example.segment_ids]),
            "attention_mask": cls._int64_feature([int(x) for x in example.attention_mask]),
            "pos_input_ids": cls._int64_feature([int(x) for x in example.pos_input_ids]),
            "pos_segment_ids": cls._int64_feature([int(x) for x in example.pos_segment_ids]),
            "pos_attention_mask": cls._int64_feature([int(x) for x in example.pos_attention_mask]),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    @classmethod
    def _filter(cls, dataset, max_sequence_length=512, **kwargs):
        dataset = dataset.filter(
            lambda a, b, c, e, f, g: tf.logical_and(
                tf.size(a) <= max_sequence_length, tf.size(e) <= max_sequence_length
            )
        )
        return dataset

    @classmethod
    def _bucket_padding(
        cls,
        dataset,
        batch_size=32,
        pad_id=0,
        bucket_boundaries=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        bucket_batch_sizes=None,
        drop_remainder=False,
        **kwargs
    ):
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        dataset = cls._bucketing(
            dataset,
            element_length_func=lambda a, b, c, e, f, g: tf.maximum(tf.size(a), tf.size(e)),
            padded_shapes=([None,], [None,], [None,], [None,], [None,], [None,]),
            padding_values=(pad_id, pad_id, pad_id, pad_id, pad_id, pad_id),
            batch_size=batch_size,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            drop_remainder=drop_remainder,
            **kwargs,
        )
        # fmt: on
        return dataset

    @classmethod
    def _batch_padding(cls, dataset, batch_size=32, pad_id=0, drop_remainder=False, **kwargs):
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([None,], [None,], [None,], [None,], [None,], [None,]),
            padding_values=(pad_id, pad_id, pad_id, pad_id, pad_id, pad_id),
            drop_remainder=drop_remainder,
        )
        # fmt: on
        return dataset

    @classmethod
    def _fixed_padding(cls, dataset, batch_size=32, pad_id=0, max_sequence_length=512, drop_remainder=False, **kwargs):
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        maxlen = tf.constant(max_sequence_length, dtype=tf.int32)
        # fmt: off
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([maxlen,], [maxlen,], [maxlen,], [maxlen,], [maxlen,], [maxlen,]),
            padding_values=(pad_id, pad_id, pad_id, pad_id, pad_id, pad_id),
            drop_remainder=drop_remainder,
        )
        # fmt: on
        return dataset

    @classmethod
    def _to_dict(cls, dataset, to_dict=True, **kwargs):
        if not to_dict:
            return dataset
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


class DataPipeForHardNegativeSimCSE(AbstractDataPipe):
    """Dataset builder for hard negavtive SimCSE"""

    @classmethod
    def _dataset_from_jsonl_files(cls, input_files, vocab_file, **kwargs) -> DatasetForHardNegativeSimCSE:
        return DatasetForHardNegativeSimCSE(input_files, vocab_file, **kwargs)

    @classmethod
    def _transform_examples_to_dataset(cls, examples, **kwargs) -> tf.data.Dataset:
        """Zip examples to dataset"""

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
    def _transform_example_to_tfrecord(cls, example, **kwargs) -> tf.train.Example:
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

    @classmethod
    def _parse_tfrecord_dataset(cls, dataset: tf.data.Dataset, **kwargs) -> tf.data.Dataset:
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
    def _filter(cls, dataset, max_sequence_length=512, **kwargs):
        dataset = dataset.filter(
            lambda a, b, c, e, f, g, h, i, j: tf.logical_and(
                tf.size(a) <= max_sequence_length,
                tf.logical_and(tf.size(e) <= max_sequence_length, tf.size(h) <= max_sequence_length),
            )
        )
        return dataset

    @classmethod
    def _bucket_padding(
        cls,
        dataset,
        batch_size=32,
        pad_id=0,
        bucket_boundaries=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        bucket_batch_sizes=None,
        drop_remainder=False,
        **kwargs
    ):
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        dataset = cls._bucketing(
            dataset,
            element_length_func=lambda a, b, c, e, f, g, h, i, j: tf.maximum(
                tf.size(a), tf.maximum(tf.size(e), tf.size(h))
            ),
            padded_shapes=([None,], [None,], [None,], [None,], [None,], [None,], [None,], [None,], [None,]),
            padding_values=(pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id),
            batch_size=batch_size,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            drop_remainder=drop_remainder,
            **kwargs,
        )
        # fmt: on
        return dataset

    @classmethod
    def _batch_padding(cls, dataset, batch_size=32, pad_id=0, drop_remainder=False, **kwargs):
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([None,], [None,], [None,], [None,], [None,], [None,], [None,], [None,], [None,]),
            padding_values=(pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id),
            drop_remainder=drop_remainder,
        )
        # fmt: on
        return dataset

    @classmethod
    def _fixed_padding(cls, dataset, batch_size=32, pad_id=0, max_sequence_length=512, drop_remainder=False, **kwargs):
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        maxlen = tf.constant(max_sequence_length, dtype=tf.int32)
        # fmt: off
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([maxlen,], [maxlen,], [maxlen,], [maxlen,], [maxlen,], [maxlen,], [maxlen,], [maxlen,], [maxlen,]),
            padding_values=(pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id),
            drop_remainder=drop_remainder,
        )
        # fmt: on
        return dataset

    @classmethod
    def _to_dict(cls, dataset, to_dict=True, **kwargs):
        if not to_dict:
            return dataset
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
