"""Dataset builder for sentiment analysis."""
import json
import os
from collections import namedtuple

import tensorflow as tf
from transformers_keras.common.char_tokenizer import BertCharTokenizer

from .abc_dataset import AbstractDataPipe, Dataset
from .qa_dataset import DataPipeForQuestionAnsweringX

ExampleForAspectTermExtraction = namedtuple(
    "ExampleForAspectTermExtraction",
    ["tokens", "input_ids", "segment_ids", "attention_mask", "start_ids", "end_ids"],
)


class DatasetForAspectTermExtraction(Dataset):
    """Dataset for ATE"""

    def __init__(
        self,
        input_files,
        vocab_file,
        do_lower_case=True,
        sequence_key="sequence",
        aspect_spans_key="aspect_spans",
        question="上文提到了那些层面？",
        **kwargs,
    ) -> None:
        super().__init__()
        self.sequence_key = sequence_key
        self.aspect_spans_key = aspect_spans_key
        self.tokenizer = BertCharTokenizer.from_file(vocab_file, do_lower_case=do_lower_case, **kwargs)
        self.instances = self._read_jsonl_files(input_files, **kwargs)

        self.question = question
        self.question_encoding = self.tokenizer.encode(self.question, add_cls=False, add_sep=True)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        instance = self.instances[index]
        sequence = instance[self.sequence_key]
        sequence_encoding = self.tokenizer.encode(sequence, add_cls=True, add_sep=True)
        question_encoding = self.question_encoding
        aspect_spans = instance[self.aspect_spans_key]
        input_length = len(sequence_encoding.ids) + len(question_encoding.ids)
        start_ids, end_ids = [0] * input_length, [0] * input_length
        tokens = sequence_encoding.tokens + question_encoding.tokens
        for span in aspect_spans:
            # start + 1: [CLS] offset
            start, end = span[0] + 1, span[1]
            assert "".join(tokens[start : end + 1]).lower() == str(sequence[span[0] : span[1]]).lower()
            start_ids[start] = 1
            end_ids[end] = 1
        example = ExampleForAspectTermExtraction(
            tokens=tokens,
            input_ids=sequence_encoding.ids + question_encoding.ids,
            segment_ids=[0] * len(sequence_encoding.ids) + [1] * len(question_encoding.ids),
            attention_mask=[1] * (len(sequence_encoding.ids) + len(question_encoding.ids)),
            start_ids=start_ids,
            end_ids=end_ids,
        )
        return example

    def _read_jsonl_files(self, input_files, **kwargs):
        instances = []
        if isinstance(input_files, str):
            input_files = [input_files]
        for f in input_files:
            if not os.path.exists(f):
                continue
            with open(f, mode="rt", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    instances.append(json.loads(line))
        return instances


class DataPipeForAspectTermExtraction(AbstractDataPipe):
    """Build dataset for aspect term extraction models."""

    @classmethod
    def _dataset_from_jsonl_files(cls, input_files, vocab_file, **kwargs) -> Dataset:
        return DatasetForAspectTermExtraction(input_files, vocab_file, **kwargs)

    @classmethod
    def _transform_examples_to_dataset(cls, examples, **kwargs) -> tf.data.Dataset:
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
                _to_dataset(x=[e.start_ids for e in examples], dtype=tf.int32),
                _to_dataset(x=[e.end_ids for e in examples], dtype=tf.int32),
            )
        )
        return dataset

    @classmethod
    def _transform_example_to_tfrecord(cls, example, **kwargs) -> tf.train.Example:
        feature = {
            "input_ids": cls._int64_feature([int(x) for x in example.input_ids]),
            "segment_ids": cls._int64_feature([int(x) for x in example.segment_ids]),
            "attention_mask": cls._int64_feature([int(x) for x in example.attention_mask]),
            "start": cls._int64_feature([int(x) for x in example.start_ids]),
            "end": cls._int64_feature([int(x) for x in example.end_ids]),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    @classmethod
    def _parse_tfrecord_dataset(cls, dataset, **kwargs) -> tf.data.Dataset:
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
                tf.cast(tf.sparse.to_dense(x["start"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["end"]), tf.int32),
            ),
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        return dataset

    @classmethod
    def _filter(cls, dataset, max_sequence_length=512, **kwargs):
        dataset = dataset.filter(
            lambda a, b, c, x, y: tf.size(a) <= max_sequence_length,
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
        **kwargs,
    ):
        # fmt: off
        dataset = cls._bucketing(
            dataset,
            element_length_func=lambda a, b, c, x, y: tf.size(a),
            padded_shapes=([None, ], [None, ], [None, ], [None, ], [None, ]),
            padding_values=(pad_id, pad_id, pad_id, pad_id, pad_id),
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
            padded_shapes=([None, ], [None, ], [None, ], [None, ], [None, ]),
            padding_values=(pad_id, pad_id, pad_id, pad_id, pad_id),
            drop_remainder=drop_remainder,
        )
        # fmt: on
        return dataset

    @classmethod
    def _fixed_padding(
        cls,
        dataset,
        batch_size=32,
        pad_id=0,
        max_sequence_length=512,
        drop_remainder=False,
        **kwargs,
    ):
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        maxlen = tf.constant(max_sequence_length, dtype=tf.int32)
        # fmt: off
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([maxlen, ], [maxlen, ], [maxlen, ], [maxlen, ], [maxlen, ]),
            padding_values=(pad_id, pad_id, pad_id, pad_id, pad_id),
            drop_remainder=drop_remainder,
        )
        # fmt: on
        return dataset

    @classmethod
    def _to_dict(cls, dataset, to_dict=True, **kwargs):
        if not to_dict:
            return dataset
        dataset = dataset.map(
            lambda a, b, c, x, y: ({"input_ids": a, "segment_ids": b, "attention_mask": c}, {"start": x, "end": y}),
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        return dataset


class DatasetForOpinionTermExtractionAndClassification(DataPipeForQuestionAnsweringX):
    """Dataset builder for Opinion Term Extraction and Classification"""

    pass
