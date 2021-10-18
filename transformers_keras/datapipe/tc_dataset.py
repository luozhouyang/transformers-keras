"""Dataset builder for token classification."""
import logging
import os
import re
from collections import namedtuple

import tensorflow as tf
from transformers_keras.common.char_tokenizer import BertCharTokenizer
from transformers_keras.common.label_tokenizer import LabelTokenizerForTokenClassification

from .abc_dataset import AbstractDataset

ExampleForTokenClassification = namedtuple(
    "ExampleForTokenClassification", ["tokens", "labels", "input_ids", "segment_ids", "attention_mask", "label_ids"]
)


class DatasetForTokenClassification(AbstractDataset):
    """Dataset for token classification."""

    @classmethod
    def from_conll_files(
        cls,
        input_files,
        tokenizer=None,
        vocab_file=None,
        label_tokenizer=None,
        label_vocab_file=None,
        sep="\t",
        verbose=True,
        n=5,
        **kwargs
    ):
        examples = cls.conll_to_examples(
            input_files,
            tokenizer=tokenizer,
            vocab_file=vocab_file,
            label_tokenizer=label_tokenizer,
            label_vocab_file=label_vocab_file,
            sep=sep,
            **kwargs,
        )
        dataset = cls.from_examples(examples, verbose=verbose, n=n, **kwargs)
        return dataset

    @classmethod
    def conll_to_examples(
        cls,
        input_files,
        tokenizer=None,
        vocab_file=None,
        label_tokenizer=None,
        label_vocab_file=None,
        sep="\t",
        **kwargs
    ):
        instances = cls._read_conll(input_files, sep=sep, **kwargs)
        examples = cls._parse_instances_to_examples(
            instances,
            tokenizer=tokenizer,
            vocab_file=vocab_file,
            label_tokenizer=label_tokenizer,
            label_vocab_file=label_vocab_file,
            features_key="features",
            labels_ley="labels",
            **kwargs,
        )
        return examples

    @classmethod
    def _read_conll(cls, input_files, sep="\t", **kwargs):
        instances = []
        if isinstance(input_files, str):
            input_files = [input_files]
        for f in input_files:
            if not os.path.exists(f):
                logging.warning("File %s does not exist, skip.", f)
                continue
            features, labels = [], []
            with open(f, mode="rt", encoding="utf-8") as fin:
                for line in fin:
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        instances.append({"features": features, "labels": labels})
                        features, labels = [], []
                        continue
                    parts = re.split(sep, line)
                    if len(parts) < 2:
                        raise ValueError("Invalid file data")
                    features.append(str(parts[0].strip()))
                    labels.append(str(parts[1]).strip())
            if features and labels:
                instances.append({"features": features, "labels": labels})
        logging.info("Read %d instances from CoNLL files.", len(instances))
        return instances

    @classmethod
    def _filter(cls, dataset, max_sequence_length=512, **kwargs):
        dataset = dataset.filter(lambda a, b, c, y: tf.size(a) <= max_sequence_length)
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
        return dataset

    @classmethod
    def _batch_padding(cls, dataset, batch_size=32, pad_id=0, drop_remainder=False, **kwargs):
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([None, ], [None, ], [None, ], [None, ]),
            padding_values=(pad_id, pad_id, pad_id, pad_id),
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
            padded_shapes=([maxlen, ], [maxlen, ], [maxlen, ], [maxlen, ]),
            padding_values=(pad_id, pad_id, pad_id, pad_id),
            drop_remainder=drop_remainder,
        )
        # fmt: on
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
    def _to_dict(cls, dataset, **kwargs):
        dataset = dataset.map(
            lambda a, b, c, y: ({"input_ids": a, "segment_ids": b, "attention_mask": c}, y),
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        return dataset

    @classmethod
    def _parse_instances_to_examples(
        cls,
        instances,
        tokenizer: BertCharTokenizer = None,
        vocab_file=None,
        label_tokenizer: LabelTokenizerForTokenClassification = None,
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
            label_tokenizer = LabelTokenizerForTokenClassification.from_file(
                label_vocab_file, o_token=kwargs.get("o_token", "O")
            )
        # collect examples
        examples = []
        for instance in instances:
            tokens = instance[kwargs.get("features_key", "features")]
            input_ids = [tokenizer.token_to_id(token) for token in tokens]
            labels = instance[kwargs.get("labels_key", "labels")]
            label_ids = label_tokenizer.labels_to_ids(labels, add_cls=False, add_sep=False)
            example = ExampleForTokenClassification(
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
    def _example_to_tfrecord(cls, example: ExampleForTokenClassification, **kwargs):
        feature = {
            "input_ids": cls._int64_feature([int(x) for x in example.input_ids]),
            "segment_ids": cls._int64_feature([int(x) for x in example.segment_ids]),
            "attention_mask": cls._int64_feature([int(x) for x in example.attention_mask]),
            "label_ids": cls._int64_feature([int(x) for x in example.label_ids]),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))
