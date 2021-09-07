import abc
import json
import logging

import tensorflow as tf
from tensorflow.python.util.decorator_utils import classproperty

TF_VERSION = tf.__version__


class AbstractDataset(abc.ABC):
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
    def from_examples(cls, examples, verbose=True, n=5, **kwargs):
        logging.info("Load %d examples in total.", len(examples))
        if verbose:
            cls._show_examples(examples, n=n, **kwargs)
        dataset = cls._zip_dataset(examples, **kwargs)
        dataset = cls._build_dataset(dataset, **kwargs)
        return dataset

    @classmethod
    def from_tfrecord_files(cls, input_files, **kwargs):
        dataset = cls._read_tfrecord(input_files, **kwargs)
        dataset = cls._build_dataset(dataset, **kwargs)
        return dataset

    @classmethod
    def from_jsonl_files(cls, input_files, **kwargs):
        examples = cls._read_jsonl(input_files, **kwargs)
        dataset = cls.from_examples(examples, **kwargs)
        return dataset

    @classmethod
    def jsonl_to_examples(cls, input_files, **kwargs):
        examples = cls._read_jsonl(input_files, **kwargs)
        return examples

    @classmethod
    def jsonl_to_tfrecord(cls, input_files, output_files, **kwargs):
        examples = cls.jsonl_to_examples(input_files, **kwargs)
        cls.examples_to_tfrecord(examples, output_files, **kwargs)

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
        logging.info("Finished to write %d examples to tfrecords.", len(examples))

    @abc.abstractclassmethod
    def _example_to_tfrecord(cls, example, **kwargs):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def _zip_dataset(cls, examples, **kwargs):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def _build_dataset(cls, dataset, **kwargs):
        pass

    @classmethod
    def _bucketing(
        cls,
        dataset,
        element_length_func,
        padded_shapes,
        padding_values,
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
        ), "len(bucket_batch_sizes) should equals len(bucket_doundaries) + 1"

        pad_id = tf.constant(pad_id, dtype=tf.int32)
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
    def _read_jsonl(cls, input_files, **kwargs):
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
        return cls._parse_jsonl(instances, **kwargs)

    @abc.abstractclassmethod
    def _parse_jsonl(cls, instances, **kwargs):
        raise NotImplementedError()

    @classmethod
    def _read_tfrecord(cls, input_files, **kwargs):
        if isinstance(input_files, str):
            input_files = [input_files]
        if len(input_files) == 1:
            dataset = tf.data.TFRecordDataset(input_files)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(input_files)
            dataset = dataset.interleave(
                lambda x: tf.data.TFRecordDataset(x),
                cycle_length=len(input_files),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        dataset = cls._parse_tfrecord(dataset, **kwargs)
        return dataset

    @abc.abstractclassmethod
    def _parse_tfrecord(cls, dataset, **kwargs):
        raise NotImplementedError()

    @classmethod
    def _auto_shard(cls, dataset, auto_shard_policy=None):
        if auto_shard_policy is not None:
            options = tf.data.Options()
            # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            options.experimental_distribute.auto_shard_policy = auto_shard_policy
            dataset = dataset.with_options(options)
        return dataset

    @classmethod
    def _show_examples(cls, examples, n=5, **kwargs):
        n = min(n, len(examples))
        logging.info("Showing %d examples.", n)
        for i in range(n):
            logging.info("NO.%d example: %s", i, examples[i])

    @classmethod
    def _int64_feature(cls, values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
