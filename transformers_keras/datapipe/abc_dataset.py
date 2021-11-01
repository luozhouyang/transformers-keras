import abc
import json
import logging
import os

import tensorflow as tf
from tensorflow.python.util.decorator_utils import classproperty
from tokenizers import BertWordPieceTokenizer


class Dataset(abc.ABC):
    """Data reader"""

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()


class DatasetForBert(Dataset):
    """Dataset for bert based models."""

    def __init__(self, vocab_file, do_lower_case=True, **kwargs) -> None:
        super().__init__()
        self.tokenizer = BertWordPieceTokenizer.from_file(vocab_file, lowercase=do_lower_case)

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


class AbstractDataPipe(abc.ABC):
    """Abstract dataset builder"""

    @classproperty
    def AUTOTUNE(cls):
        try:
            autotune = tf.data.experimental.AUTOTUNE
            return autotune
        except Exception as e:
            try:
                autotune = tf.data.AUTOTUNE
                return autotune
            except Exception as e:
                logging.warning("Found AUTOTUNE exception: ", e)
        return None

    @classproperty
    def bucket_by_sequence_length(cls):
        try:
            fn = tf.data.experimental.bucket_by_sequence_length
            return fn
        except Exception as e:
            try:
                fn = tf.data.bucket_by_sequence_length
                return fn
            except Exception as e:
                logging.warning("Find bucket_by_sequence_length exception: ", e)
        return None

    @classmethod
    def from_dataset(cls, dataset: Dataset, **kwargs):
        examples = [dataset[idx] for idx in range(len(dataset))]
        return cls.from_examples(examples, **kwargs)

    @classmethod
    def from_jsonl_files(cls, input_files, vocab_file, **kwargs):
        dataset = cls._dataset_from_jsonl_files(input_files, vocab_file, **kwargs)
        return cls.from_dataset(dataset, **kwargs)

    @classmethod
    def _dataset_from_jsonl_files(cls, input_files, vocab_file, **kwargs) -> Dataset:
        raise NotImplementedError()

    @classmethod
    def from_examples(cls, examples, verbose=True, n=5, **kwargs):
        logging.info("Load %d examples in total.", len(examples))
        cls._show_examples(examples, verbose=verbose, n=n, **kwargs)
        dataset = cls._transform_examples_to_dataset(examples, **kwargs)
        dataset = cls._transform(dataset, **kwargs)
        return dataset

    @classmethod
    def from_tfrecord_files(cls, input_files, **kwargs):
        dataset = cls._read_tfrecord_files(input_files, **kwargs)
        dataset = cls._parse_tfrecord_dataset(dataset, **kwargs)
        dataset = cls._transform(dataset, **kwargs)
        return dataset

    @classmethod
    def dataset_to_tfrecord(cls, dataset: Dataset, output_files, **kwargs):
        examples = [dataset[idx] for idx in range(len(dataset))]
        cls.examples_to_tfrecord(examples, output_files, **kwargs)

    @classmethod
    def jsonl_to_examples(cls, input_files, vocab_file, **kwargs):
        dataset = cls._dataset_from_jsonl_files(input_files, vocab_file, **kwargs)
        examples = [dataset[idx] for idx in range(len(dataset))]
        return examples

    @classmethod
    def examples_to_tfrecord(cls, examples, output_files, **kwargs):
        if isinstance(output_files, str):
            output_files = [output_files]
        writers = [tf.io.TFRecordWriter(f) for f in output_files]
        idx = 0
        for example in examples:
            tfrecord_example = cls._transform_example_to_tfrecord(example, **kwargs)
            writers[idx].write(tfrecord_example.SerializeToString())
            idx += 1
            idx = idx % len(writers)
        for w in writers:
            w.close()
        logging.info("Finished to write %d examples to tfrecords.", len(examples))

    @classmethod
    def jsonl_to_tfrecord(cls, input_files, output_files, tokenizer=None, vocab_file=None, **kwargs):
        examples = cls.jsonl_to_examples(input_files, tokenizer=tokenizer, vocab_file=vocab_file, **kwargs)
        cls.examples_to_tfrecord(examples, output_files, **kwargs)

    @classmethod
    def _transform_example_to_tfrecord(cls, example, **kwargs) -> tf.train.Example:
        raise NotImplementedError()

    @classmethod
    def _transform_examples_to_dataset(cls, examples, **kwargs) -> tf.data.Dataset:
        raise NotImplementedError()

    @classmethod
    def _parse_tfrecord_dataset(cls, dataset: tf.data.Dataset, **kwargs) -> tf.data.Dataset:
        raise NotImplementedError()

    @classmethod
    def _transform(
        cls,
        dataset,
        batch_size=64,
        max_sequence_length=512,
        padding="bucket",
        bucket_boundaries=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        bucket_batch_sizes=None,
        repeat=None,
        shuffle=True,
        buffer_size=1000000,
        seed=None,
        reshuffle_each_iteration=True,
        pad_id=0,
        auto_shard_policy=None,
        drop_remainder=False,
        to_dict=True,
        **kwargs,
    ):
        dataset = cls._filter(dataset, max_sequence_length=max_sequence_length, **kwargs)
        dataset = cls._repeat(dataset, repeat=repeat, **kwargs)
        dataset = cls._shuffle(
            dataset,
            shuffle=shuffle,
            buffer_size=buffer_size,
            seed=seed,
            reshuffle_each_iteration=reshuffle_each_iteration,
            **kwargs,
        )
        assert padding in ["bucket", "batch", "fixed"], "Invalid padding: " + str(padding)
        if padding == "bucket":
            dataset = cls._bucket_padding(
                dataset,
                pad_id=pad_id,
                batch_size=batch_size,
                bucket_boundaries=bucket_boundaries,
                bucket_batch_sizes=bucket_batch_sizes,
                drop_remainder=drop_remainder,
                **kwargs,
            )
        elif padding == "batch":
            dataset = cls._batch_padding(
                dataset,
                batch_size=batch_size,
                pad_id=pad_id,
                drop_remainder=drop_remainder,
                **kwargs,
            )
        elif padding == "fixed":
            dataset = cls._fixed_padding(
                dataset,
                batch_size=batch_size,
                pad_id=pad_id,
                max_sequence_length=max_sequence_length,
                drop_remainder=drop_remainder,
                **kwargs,
            )
        dataset = dataset.prefetch(cls.AUTOTUNE)
        dataset = cls._to_dict(dataset, to_dict=to_dict, **kwargs)
        dataset = cls._auto_shard(dataset, auto_shard_policy=auto_shard_policy, **kwargs)
        return dataset

    @classmethod
    def _filter(cls, dataset, max_sequence_length=512, **kwargs):
        raise NotImplementedError()

    @classmethod
    def _shuffle(cls, dataset, shuffle=True, buffer_size=100000, seed=None, reshuffle_each_iteration=True, **kwargs):
        if not shuffle:
            return dataset
        return dataset.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)

    @classmethod
    def _repeat(cls, dataset, repeat=None, **kwargs):
        if repeat is None:
            return dataset
        dataset = dataset.repeat(repeat)
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
        raise NotImplementedError()

    @classmethod
    def _batch_padding(cls, dataset, batch_size=32, pad_id=0, drop_remainder=False, **kwargs):
        raise NotImplementedError()

    @classmethod
    def _fixed_padding(cls, dataset, batch_size=32, pad_id=0, max_sequence_length=512, drop_remainder=False, **kwargs):
        raise NotImplementedError()

    @classmethod
    def _to_dict(cls, dataset, to_dict=True, **kwargs):
        raise NotImplementedError()

    @classmethod
    def _bucketing(
        cls,
        dataset,
        element_length_func,
        padded_shapes,
        padding_values,
        batch_size=64,
        bucket_boundaries=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        bucket_batch_sizes=None,
        drop_remainder=False,
        **kwargs,
    ):
        if bucket_batch_sizes is None:
            bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
        assert (
            len(bucket_batch_sizes) == len(bucket_boundaries) + 1
        ), "len(bucket_batch_sizes) should equals len(bucket_doundaries) + 1"

        # fmt: off
        dataset = dataset.apply(cls.bucket_by_sequence_length(
            element_length_func=element_length_func,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            drop_remainder=drop_remainder,
        )).prefetch(cls.AUTOTUNE)
        # fmt: on
        return dataset

    @classmethod
    def _read_jsonl_files(cls, input_files, **kwargs):
        if isinstance(input_files, str):
            input_files = [input_files]
        instances = []
        for f in input_files:
            with open(f, mode="rt", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    instance = json.loads(line)
                    instances.append(instance)
        logging.info("Collected %d instances in total.", len(instances))
        return instances

    @classmethod
    def _parse_instances_to_examples(cls, instances, tokenizer=None, vocab_file=None, **kwargs):
        raise NotImplementedError()

    @classmethod
    def _read_tfrecord_files(cls, input_files, **kwargs):
        if isinstance(input_files, str):
            input_files = [input_files]
        if len(input_files) == 1:
            dataset = tf.data.TFRecordDataset(input_files)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(input_files)
            dataset = dataset.interleave(
                lambda x: tf.data.TFRecordDataset(x),
                cycle_length=len(input_files),
                num_parallel_calls=cls.AUTOTUNE,
            )
        return dataset

    @classmethod
    def _auto_shard(cls, dataset, auto_shard_policy=None, **kwargs):
        if auto_shard_policy is not None:
            options = tf.data.Options()
            # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            options.experimental_distribute.auto_shard_policy = auto_shard_policy
            dataset = dataset.with_options(options)
        return dataset

    @classmethod
    def _show_examples(cls, examples, verbose=True, n=5, **kwargs):
        if not verbose:
            return
        n = min(n, len(examples))
        logging.info("Showing %d examples.", n)
        for i in range(n):
            logging.info("NO.%d example: %s", i, examples[i])

    @classmethod
    def _int64_feature(cls, values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
