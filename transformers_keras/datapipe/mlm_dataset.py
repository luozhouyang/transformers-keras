"""Dataset builder for masked language models."""
import abc
import json
import os
import random
from collections import namedtuple

import tensorflow as tf
from tokenizers import BertWordPieceTokenizer

from .abc_dataset import AbstractDataPipe, Dataset, DatasetForBert

ExampleForMaskedLanguageModel = namedtuple(
    "ExampleForMaskedLanguageModel",
    ["tokens", "input_ids", "segment_ids", "attention_mask", "masked_ids", "masked_pos"],
)

ResultForMasking = namedtuple("ResultForMasking", ["origin_tokens", "masked_tokens", "masked_indexes"])


class AbstractMaskingStrategy(abc.ABC):
    """Abstract masking strategy"""

    @abc.abstractmethod
    def __call__(self, tokens, **kwargs) -> ResultForMasking:
        raise NotImplementedError()


class WholeWordMask(AbstractMaskingStrategy):
    """Default masking strategy from BERT."""

    def __init__(
        self, vocabs, change_prob=0.15, mask_prob=0.8, rand_prob=0.1, keep_prob=0.1, max_predictions=20, **kwargs
    ):
        self.vocabs = vocabs
        self.change_prob = change_prob
        self.mask_prob = mask_prob / (mask_prob + rand_prob + keep_prob)
        self.rand_prob = rand_prob / (mask_prob + rand_prob + keep_prob)
        self.keep_prob = keep_prob / (mask_prob + rand_prob + keep_prob)
        self.max_predictions = max_predictions

    def __call__(self, tokens, max_sequence_length=512, **kwargs) -> ResultForMasking:
        tokens = self._truncate_sequence(tokens, max_sequence_length - 2)
        if not tokens:
            return None
        num_to_predict = min(self.max_predictions, max(1, round(self.change_prob * len(tokens))))
        cand_indexes = self._collect_candidates(tokens)
        # copy original tokens
        masked_tokens = [x for x in tokens]
        masked_indexes = [0] * len(tokens)
        for piece_indexes in cand_indexes:
            if sum(masked_indexes) >= num_to_predict:
                break
            if sum(masked_indexes) + len(piece_indexes) > num_to_predict:
                continue
            if any(masked_indexes[idx] == 1 for idx in piece_indexes):
                continue
            for index in piece_indexes:
                masked_indexes[index] = 1
                masked_tokens[index] = self._masking_tokens(index, tokens, self.vocabs)

        # add special tokens
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        masked_tokens = ["[CLS]"] + masked_tokens + ["[SEP]"]
        masked_indexes = [0] + masked_indexes + [0]
        assert len(tokens) == len(masked_tokens) == len(masked_indexes)
        return ResultForMasking(origin_tokens=tokens, masked_tokens=masked_tokens, masked_indexes=masked_indexes)

    def _masking_tokens(self, index, tokens, vocabs, **kwargs):
        # 80% of the time, replace with [MASK]
        if random.random() < self.mask_prob:
            return "[MASK]"
        # 10% of the time, keep original
        p = self.rand_prob / (self.rand_prob + self.keep_prob)
        if random.random() < p:
            return tokens[index]
        # 10% of the time, replace with random word
        masked_token = vocabs[random.randint(0, len(vocabs) - 1)]
        return masked_token

    def _collect_candidates(self, tokens):
        cand_indexes = [[]]
        for idx, token in enumerate(tokens):
            if cand_indexes and token.startswith("##"):
                cand_indexes[-1].append(idx)
                continue
            cand_indexes.append([idx])
        random.shuffle(cand_indexes)
        return cand_indexes

    def _truncate_sequence(self, tokens, max_tokens=512, **kwargs):
        while len(tokens) > max_tokens:
            if len(tokens) > max_tokens:
                tokens.pop(0)
                # truncate whole world
                while tokens and tokens[0].startswith("##"):
                    tokens.pop(0)
            if len(tokens) > max_tokens:
                while tokens and tokens[-1].startswith("##"):
                    tokens.pop()
                if tokens:
                    tokens.pop()
        return tokens


class DatasetForMaskedLanguageModel(DatasetForBert):
    """Default dataset for masked lm"""

    def __init__(
        self,
        input_files,
        vocab_file,
        masking: AbstractMaskingStrategy = None,
        chage_prob=0.15,
        mask_prob=0.8,
        rand_prob=0.1,
        keep_prob=0.1,
        max_predictions=20,
        add_special_tokens=False,
        sequence_key="sequence",
        max_sequence_length=512,
        **kwargs
    ):
        """Default dataset for masked language model.

        Args:
            input_files: List of jsonl files
            masking: Callable or instance of AbstractMaskingStrategy, to masking tokens
        """
        super().__init__(vocab_file, **kwargs)
        self.sequence_key = sequence_key
        self.max_sequence_length = max_sequence_length
        self.vocabs = list(self.tokenizer.get_vocab().keys())
        self.add_special_tokens = add_special_tokens

        self.masking = masking or WholeWordMask(
            vocabs=self.vocabs,
            change_prob=chage_prob,
            mask_prob=mask_prob,
            rand_prob=rand_prob,
            keep_prob=keep_prob,
            max_predictions=max_predictions,
        )
        self.instances = self._read_jsonl_files(input_files, **kwargs)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> ExampleForMaskedLanguageModel:
        instance = self.instances[index]
        encoding = self.tokenizer.encode(instance["sequence"], add_special_tokens=self.add_special_tokens)
        results = self.masking(
            tokens=encoding.tokens,
            max_sequence_length=self.max_sequence_length,
        )
        origin_tokens, masked_tokens = results.origin_tokens, results.masked_tokens
        return ExampleForMaskedLanguageModel(
            tokens=masked_tokens,
            input_ids=[self.tokenizer.token_to_id(x) for x in masked_tokens],
            segment_ids=[0] * len(masked_tokens),
            attention_mask=[1] * len(masked_tokens),
            masked_ids=[self.tokenizer.token_to_id(x) for x in origin_tokens],
            masked_pos=results.masked_indexes,
        )


class DataPipeForMaskedLanguageModel(AbstractDataPipe):
    """Dataset builder for masked language model."""

    @classmethod
    def _dataset_from_jsonl_files(
        cls, input_files, vocab_file, masking: AbstractMaskingStrategy = None, **kwargs
    ) -> Dataset:
        dataset = DatasetForMaskedLanguageModel(
            input_files=input_files,
            vocab_file=vocab_file,
            masking=masking,
            **kwargs,
        )
        return dataset

    @classmethod
    def _transform_examples_to_dataset(cls, examples, **kwargs):
        """Trasnform examples to tf.data.Dataset instance"""

        def _to_dataset(x, dtype=tf.int32):
            x = tf.ragged.constant(x, dtype=dtype)
            d = tf.data.Dataset.from_tensor_slices(x)
            d = d.map(lambda x: x)
            return d

        # conver examples to dataset
        dataset = tf.data.Dataset.zip(
            (
                _to_dataset([e.input_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.segment_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.attention_mask for e in examples], dtype=tf.int32),
                _to_dataset([e.masked_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.masked_pos for e in examples], dtype=tf.int32),
            )
        )
        return dataset

    @classmethod
    def _transform_example_to_tfrecord(cls, example, **kwargs):
        feature = {
            "input_ids": cls._int64_feature([int(x) for x in example.input_ids]),
            "segment_ids": cls._int64_feature([int(x) for x in example.segment_ids]),
            "attention_mask": cls._int64_feature([int(x) for x in example.attention_mask]),
            "masked_ids": cls._int64_feature([int(x) for x in example.masked_ids]),
            "masked_pos": cls._int64_feature([int(x) for x in example.masked_pos]),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    @classmethod
    def _parse_tfrecord_dataset(cls, dataset, **kwargs):
        features = {
            "input_ids": tf.io.VarLenFeature(tf.int64),
            "segment_ids": tf.io.VarLenFeature(tf.int64),
            "attention_mask": tf.io.VarLenFeature(tf.int64),
            "masked_ids": tf.io.VarLenFeature(tf.int64),
            "masked_pos": tf.io.VarLenFeature(tf.int64),
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
                tf.cast(tf.sparse.to_dense(x["masked_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["masked_pos"]), tf.int32),
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
        bucket_boundaries=[50, 100, 150, 200, 250, 200, 250, 400, 450],
        bucket_batch_sizes=None,
        drop_remainder=False,
        **kwargs
    ):
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        dataset = cls._bucketing(
            dataset,
            element_length_func=lambda a, b, c, x, y: tf.size(a),
            padded_shapes=([None,], [None,], [None,], [None,], [None,]),
            padding_values=(pad_id, pad_id, pad_id, pad_id, pad_id),
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
            padded_shapes=([None,], [None,], [None,], [None,], [None,]),
            padding_values=(pad_id, pad_id, pad_id, pad_id, pad_id),
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
            padded_shapes=([maxlen,], [maxlen,], [maxlen,], [maxlen,], [maxlen,]),
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
            lambda a, b, c, x, y: (
                {"input_ids": a, "segment_ids": b, "attention_mask": c},
                {"masked_ids": x, "masked_pos": y},
            ),
            num_parallel_calls=cls.AUTOTUNE,
        ).prefetch(cls.AUTOTUNE)
        return dataset
