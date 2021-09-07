from collections import namedtuple

import tensorflow as tf
from tokenizers import BertWordPieceTokenizer
from transformers_keras.dataset_utils import AbstractDataset
from transformers_keras.tokenizers.char_tokenizer import BertCharTokenizer

from .tokenizer import TokenClassificationLabelTokenizer

TokenClassificationExample = namedtuple(
    "TokenClassificationExample", ["tokens", "labels", "input_ids", "segment_ids", "attention_mask", "label_ids"]
)


class TokenClassificationDataset(AbstractDataset):
    """Dataset for token classification."""

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
        dataset = dataset.filter(lambda a, b, c, y: tf.size(a) <= max_sequence_length)
        if repeat is not None:
            dataset = dataset.repeat(repeat)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)
        # fmt: off
        dataset = cls._bucketing(
            dataset,
            element_length_func=lambda a, b, c, y: tf.size(a),
            padded_shapes=([None, ], [None, ], [None, ], [None, ]),
            padding_values=(pad_id, pad_id, pad_id, pad_id),
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
    def _to_dict(cls, dataset):
        dataset = dataset.map(
            lambda a, b, c, y: ({"input_ids": a, "segment_ids": b, "attention_mask": c}, y),
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        return dataset

    @classmethod
    def _parse_jsonl(
        cls,
        instances,
        tokenizer: BertCharTokenizer = None,
        vocab_file=None,
        label_tokenizer: TokenClassificationLabelTokenizer = None,
        label_vocab_file=None,
        **kwargs
    ):
        assert tokenizer or vocab_file, "`tokenizer` or `vocab_file` must be provided."
        if tokenizer is None:
            tokenizer = BertCharTokenizer.from_file(
                vocab_file,
                do_lower_case=kwargs.get("do_lower_case", True),
            )
        assert label_tokenizer or label_vocab_file, "`label_tokenizer` or `label_vocab_file` must be provided."
        if label_tokenizer is None:
            label_tokenizer = TokenClassificationLabelTokenizer.from_file(label_vocab_file, o_token=kwargs.get("o_token", "O"))
        # collect examples
        examples = []
        for instance in instances:
            tokens = instance[kwargs.get("features_key", "features")]
            input_ids = [tokenizer.token_to_id(token) for token in tokens]
            labels = instance[kwargs.get("labels_key", "labels")]
            label_ids = label_tokenizer.labels_to_ids(labels, add_cls=False, add_sep=False)
            example = TokenClassificationExample(
                tokens=tokens,
                labels=labels,
                input_ids=input_ids,
                segment_ids=[0] * len(input_ids),
                attention_mask=[1] * len(input_ids),
                label_ids=label_ids,
            )
            examples.append(example)
        return examples

    @classmethod
    def _parse_tfrecord(cls, dataset, **kwargs):
        features = {
            "input_ids": tf.io.VarLenFeature(tf.int64),
            "segment_ids": tf.io.VarLenFeature(tf.int64),
            "attention_mask": tf.io.VarLenFeature(tf.int64),
            "label_ids": tf.io.VarLenFeature(tf.int64),
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
                tf.cast(tf.sparse.to_dense(x["label_ids"]), tf.int32),
            ),
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        return dataset

    @classmethod
    def _example_to_tfrecord(cls, example: TokenClassificationExample, **kwargs):
        feature = {
            "input_ids": cls._int64_feature([int(x) for x in example.input_ids]),
            "segment_ids": cls._int64_feature([int(x) for x in example.segment_ids]),
            "attention_mask": cls._int64_feature([int(x) for x in example.attention_mask]),
            "label_ids": cls._int64_feature([int(x) for x in example.label_ids]),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))
