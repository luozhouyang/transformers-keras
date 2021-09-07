import logging
from collections import namedtuple
from typing import List

import tensorflow as tf
from transformers_keras.dataset_utils import AbstractDataset
from transformers_keras.question_answering.dataset import QuestionAnsweringXDataset
from transformers_keras.tokenizers.char_tokenizer import BertCharTokenizer

AspectTermExtractionExample = namedtuple(
    "AspectTermExtractionExample",
    ["tokens", "input_ids", "segment_ids", "attention_mask", "start_ids", "end_ids"],
)


class AspectTermExtractionDataset(AbstractDataset):
    """Build dataset for aspect term extraction models."""

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
        dataset = dataset.filter(
            lambda a, b, c, x, y: tf.size(a) <= max_sequence_length,
        )
        if repeat is not None:
            dataset = dataset.repeat(repeat)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)
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
        dataset = cls._to_dict(dataset)
        dataset = cls._auto_shard(dataset, auto_shard_policy=auto_shard_policy)
        return dataset

    @classmethod
    def _to_dict(cls, dataset):
        dataset = dataset.map(
            lambda a, b, c, x, y: ({"input_ids": a, "segment_ids": b, "attention_mask": c}, {"start": x, "end": y}),
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        return dataset

    @classmethod
    def _zip_dataset(cls, examples: List[AspectTermExtractionExample], **kwargs):
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
    def _parse_jsonl(cls, instances, tokenizer: BertCharTokenizer = None, vocab_file=None, question="文中提到了哪些层面？", **kwargs):
        assert tokenizer or vocab_file, "`tokenizer` or `vocab_file` must be provided."
        if tokenizer is None:
            tokenizer = BertCharTokenizer.from_file(
                vocab_file,
                do_lower_case=kwargs.get("do_lower_case", True),
            )
        examples = []
        for instance in instances:
            sequence = instance[kwargs.get("sequence_key", "sequence")]
            sequence_encoding = tokenizer.encode(sequence, add_cls=True, add_sep=True)
            question_encoding = tokenizer.encode(question, add_cls=False, add_sep=True)
            aspect_spans = instance[kwargs.get("aspect_spans_key", "aspect_spans")]
            input_length = len(sequence_encoding.ids) + len(question_encoding.ids)
            start_ids, end_ids = [0] * input_length, [0] * input_length
            tokens = sequence_encoding.tokens + question_encoding.tokens
            for span in aspect_spans:
                # start + 1: [CLS] offset
                start, end = span[0] + 1, span[1]
                assert "".join(tokens[start : end + 1]).lower() == str(sequence[span[0] : span[1]]).lower()
                start_ids[start] = 1
                end_ids[end] = 1
            example = AspectTermExtractionExample(
                tokens=tokens,
                input_ids=sequence_encoding.ids + question_encoding.ids,
                segment_ids=[0] * len(sequence_encoding.ids) + [1] * len(question_encoding.ids),
                attention_mask=[1] * (len(sequence_encoding.ids) + len(question_encoding.ids)),
                start_ids=start_ids,
                end_ids=end_ids,
            )
            examples.append(example)
        logging.info("Collected %d examples in total.", len(examples))
        return examples

    @classmethod
    def _parse_tfrecord(cls, dataset, **kwargs):
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
    def _example_to_tfrecord(cls, example: AspectTermExtractionExample, **kwargs):
        feature = {
            "input_ids": cls._int64_feature([int(x) for x in example.input_ids]),
            "segment_ids": cls._int64_feature([int(x) for x in example.segment_ids]),
            "attention_mask": cls._int64_feature([int(x) for x in example.attention_mask]),
            "start": cls._int64_feature([int(x) for x in example.start_ids]),
            "end": cls._int64_feature([int(x) for x in example.end_ids]),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))


class OpinionTermExtractionAndClassificationDataset(QuestionAnsweringXDataset):
    """Dataset builder for Opinion Term Extraction and Classification"""

    pass
