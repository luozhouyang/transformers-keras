import re
from collections import namedtuple

import tensorflow as tf
from tensorflow.core import example

from .abc_dataset import AbstractDataPipe, DatasetForBert

ExampleForQuestionAnswering = namedtuple(
    "ExampleForQuestionAnswering", ["tokens", "input_ids", "segment_ids", "attention_mask", "start", "end"]
)
ExampleForQuestionAnsweringX = namedtuple(
    "ExampleForQuestionAnsweringX",
    ["tokens", "input_ids", "segment_ids", "attention_mask", "start", "end", "class_id"],
)


class PredictioDatasetForQuestionAnswering(DatasetForBert):
    """Question answering dataset for prediction"""

    def __init__(
        self, input_files, vocab_file, add_special_tokens=True, context_key="context", question_key="question", **kwargs
    ) -> None:
        super().__init__(vocab_file, **kwargs)
        self.context_key = context_key
        self.question_key = question_key
        self.instances = self._read_jsonl_files(input_files, **kwargs)
        self.add_special_tokens = add_special_tokens

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        instance = self.instances[index]
        context, question = instance[self.context_key], instance[self.question_key]
        encoding = self.tokenizer.encode(context, question, add_special_tokens=self.add_special_tokens)
        return ExampleForQuestionAnswering(
            tokens=encoding.tokens,
            input_ids=encoding.ids,
            segment_ids=encoding.type_ids,
            attention_mask=encoding.attention_mask,
            start=None,
            end=None,
        )


class DatasetForQuestionAnswering(DatasetForBert):
    """Dataset for qa"""

    def __init__(
        self, input_files, vocab_file, context_key="context", question_key="question", answers_key="answer", **kwargs
    ) -> None:
        super().__init__(vocab_file, **kwargs)
        self.context_key = context_key
        self.question_key = question_key
        self.answers_key = answers_key
        self.instances = self._read_jsonl_files(input_files, **kwargs)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> ExampleForQuestionAnswering:
        instance = self.instances[index]
        context = instance[self.context_key]
        question = instance[self.question_key]
        answers = instance[self.answers_key]
        if isinstance(answers, str):
            answers = [answers]
        answer = answers[0]
        example = self._parse_example(context, question, answer)
        return example

    def _find_answer_span(self, context, answer):
        for m in re.finditer(re.escape(answer), context, re.IGNORECASE):
            start, end = m.span()
            return start, end
        return 0, 0

    def _parse_example(self, context, question, answer, **kwargs):
        start_char_idx, end_char_idx = self._find_answer_span(context, answer)
        if end_char_idx <= start_char_idx:
            return None
        # Mark the character indexes in context that are in answer
        is_char_in_ans = [0] * len(context)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1
        context_encoding = self.tokenizer.encode(context)
        # Find tokens that were created from answer characters
        ans_token_idx = []
        for idx, (start_char_idx, end_char_idx) in enumerate(context_encoding.offsets):
            if sum(is_char_in_ans[start_char_idx:end_char_idx]) > 0:
                ans_token_idx.append(idx)
        if not ans_token_idx:
            return None
        start_token_idx, end_token_idx = ans_token_idx[0], ans_token_idx[-1]
        question_encoding = self.tokenizer.encode(question)
        input_ids = context_encoding.ids + question_encoding.ids[1:]
        segment_ids = [0] * len(context_encoding.type_ids) + [1] * len(question_encoding.type_ids[1:])
        attention_mask = [1] * len(context_encoding.attention_mask + question_encoding.attention_mask[1:])
        assert len(input_ids) == len(segment_ids), "input_ids length:{} VS segment_ids length: {}".format(
            len(input_ids), len(segment_ids)
        )
        assert len(input_ids) == len(attention_mask), "input_ids length:{} VS attention_mask length: {}".format(
            len(input_ids), len(attention_mask)
        )

        example = ExampleForQuestionAnswering(
            tokens=context_encoding.tokens + question_encoding.tokens[1:],
            input_ids=input_ids,
            segment_ids=segment_ids,
            attention_mask=attention_mask,
            start=start_token_idx,
            end=end_token_idx,
        )
        return example


class DatasetForQuestionAnsweringX(DatasetForQuestionAnswering):
    """Dataset for qa with classification head"""

    def __init__(self, input_files, vocab_file, class_key="class", **kwargs) -> None:
        super().__init__(input_files, vocab_file, **kwargs)
        self.class_key = class_key

    def __getitem__(self, index) -> ExampleForQuestionAnsweringX:
        instance = self.instances[index]
        context = instance[self.context_key]
        question = instance[self.question_key]
        answers = instance[self.answers_key]
        if isinstance(answers, str):
            answers = [answers]
        answer = answers[0]
        class_id = int(instance[self.class_key])
        example = self._parse_example(context, question, answer, class_id=class_id)
        return example

    def _parse_example(self, context, question, answer, class_id=None, **kwargs):
        assert class_id is not None
        context_encoding = self.tokenizer.encode(context)
        question_encoding = self.tokenizer.encode(question)
        start_char_idx, end_char_idx = self._find_answer_span(context, answer)
        # invalid answer index
        if end_char_idx <= start_char_idx:
            raise ValueError("end index <= start index.")

        # Mark the character indexes in context that are in answer
        is_char_in_ans = [0] * len(context)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1
        # Find tokens that were created from answer characters
        ans_token_idx = []
        for idx, (start_char_idx, end_char_idx) in enumerate(context_encoding.offsets):
            if sum(is_char_in_ans[start_char_idx:end_char_idx]) > 0:
                ans_token_idx.append(idx)
        # set token index of start & end
        start_token_idx = ans_token_idx[0] if ans_token_idx else 0
        end_token_idx = ans_token_idx[-1] if ans_token_idx else 0
        # build example
        tokens = context_encoding.tokens + question_encoding.tokens[1:]
        input_ids = context_encoding.ids + question_encoding.ids[1:]
        segment_ids = [0] * len(context_encoding.type_ids) + [1] * len(context_encoding.type_ids[1:])
        attention_mask = [1] * len(context_encoding.attention_mask + question_encoding.attention_mask[1:])
        return ExampleForQuestionAnsweringX(
            tokens=tokens,
            input_ids=input_ids,
            segment_ids=segment_ids,
            attention_mask=attention_mask,
            start=start_token_idx,
            end=end_token_idx,
            class_id=class_id,
        )


class DataPipeForQuestionAnswering(AbstractDataPipe):
    """Dataset builder for question answering models."""

    @classmethod
    def _dataset_from_jsonl_files(cls, input_files, vocab_file, **kwargs) -> DatasetForQuestionAnswering:
        return DatasetForQuestionAnswering(input_files, vocab_file, **kwargs)

    @classmethod
    def _transform_examples_to_dataset(cls, examples, **kwargs):
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
    def _transform_example_to_tfrecord(cls, example, **kwargs):
        feature = {
            "input_ids": cls._int64_feature([int(x) for x in example.input_ids]),
            "segment_ids": cls._int64_feature([int(x) for x in example.segment_ids]),
            "attention_mask": cls._int64_feature([int(x) for x in example.attention_mask]),
            "start": cls._int64_feature([int(example.start)]),
            "end": cls._int64_feature([int(example.end)]),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    @classmethod
    def _parse_tfrecord_dataset(cls, dataset, **kwargs):
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
                tf.cast(tf.squeeze(tf.sparse.to_dense(x["start"])), tf.int32),
                tf.cast(tf.squeeze(tf.sparse.to_dense(x["end"])), tf.int32),
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
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        dataset = cls._bucketing(
            dataset,
            element_length_func=lambda a, b, c, x, y: tf.size(a),
            padded_shapes=([None, ], [None, ], [None, ], [], []),
            padding_values=(pad_id, pad_id, pad_id, None, None),
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
            padded_shapes=([None, ], [None, ], [None, ], [], []),
            padding_values=(pad_id, pad_id, pad_id, None, None),
            drop_remainder=drop_remainder,
        ).prefetch(cls.AUTOTUNE)
        # fmt: on
        return dataset

    @classmethod
    def _fixed_padding(cls, dataset, batch_size=32, pad_id=0, max_sequence_length=512, drop_remainder=False, **kwargs):
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        maxlen = tf.constant(max_sequence_length, dtype=tf.int32)
        # fmt: off
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([maxlen, ], [maxlen, ], [maxlen, ], [], []),
            padding_values=(pad_id, pad_id, pad_id, None, None),
            drop_remainder=drop_remainder,
        ).prefetch(cls.AUTOTUNE)
        # fmt: on
        return dataset

    @classmethod
    def _to_dict(cls, dataset, to_dict=True, **kwargs):
        if not to_dict:
            return dataset
        dataset = dataset.map(
            lambda a, b, c, x, y: ({"input_ids": a, "segment_ids": b, "attention_mask": c}, {"head": x, "tail": y}),
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        return dataset


class DataPipeForQuestionAnsweringX(AbstractDataPipe):
    """Dataset builder for question answering models."""

    @classmethod
    def _dataset_from_jsonl_files(cls, input_files, vocab_file, **kwargs) -> DatasetForQuestionAnsweringX:
        return DatasetForQuestionAnsweringX(input_files, vocab_file, **kwargs)

    @classmethod
    def _transform_examples_to_dataset(cls, examples, **kwargs):
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
                _to_dataset(x=[e.class_id for e in examples], dtype=tf.int32),
            )
        )
        return dataset

    @classmethod
    def _transform_example_to_tfrecord(cls, example, **kwargs):
        feature = {
            "input_ids": cls._int64_feature([int(x) for x in example.input_ids]),
            "segment_ids": cls._int64_feature([int(x) for x in example.segment_ids]),
            "attention_mask": cls._int64_feature([int(x) for x in example.attention_mask]),
            "start": cls._int64_feature([int(example.start)]),
            "end": cls._int64_feature([int(example.end)]),
            "class_id": cls._int64_feature([int(example.class_id)]),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    @classmethod
    def _parse_tfrecord_dataset(cls, dataset, **kwargs):
        features = {
            "input_ids": tf.io.VarLenFeature(tf.int64),
            "segment_ids": tf.io.VarLenFeature(tf.int64),
            "attention_mask": tf.io.VarLenFeature(tf.int64),
            "start": tf.io.VarLenFeature(tf.int64),
            "end": tf.io.VarLenFeature(tf.int64),
            "class_id": tf.io.VarLenFeature(tf.int64),
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
                tf.cast(tf.squeeze(tf.sparse.to_dense(x["start"])), tf.int32),
                tf.cast(tf.squeeze(tf.sparse.to_dense(x["end"])), tf.int32),
                tf.cast(tf.squeeze(tf.sparse.to_dense(x["class_id"])), tf.int32),
            ),
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        return dataset

    @classmethod
    def _filter(cls, dataset, max_sequence_length=512, **kwargs):
        dataset = dataset.filter(
            lambda a, b, c, x, y, z: tf.size(a) <= max_sequence_length,
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
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        dataset = cls._bucketing(
            dataset,
            element_length_func=lambda a, b, c, x, y, z: tf.size(a),
            padded_shapes=([None, ], [None, ], [None, ], [], [], []),
            padding_values=(pad_id, pad_id, pad_id, None, None, None),
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
            padded_shapes=([None, ], [None, ], [None, ], [], [], []),
            padding_values=(pad_id, pad_id, pad_id, None, None, None),
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
            padded_shapes=([maxlen, ], [maxlen, ], [maxlen, ], [], [], []),
            padding_values=(pad_id, pad_id, pad_id, None, None, None),
            drop_remainder=drop_remainder,
        )
        # fmt: on
        return dataset

    @classmethod
    def _to_dict(cls, dataset, to_dict=True, **kwargs):
        if not to_dict:
            return dataset
        dataset = dataset.map(
            lambda a, b, c, x, y, z: (
                {"input_ids": a, "segment_ids": b, "attention_mask": c},
                {"head": x, "tail": y, "class": z},
            ),
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        return dataset
