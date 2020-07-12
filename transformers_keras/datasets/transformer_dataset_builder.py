import tensorflow as tf
from naivenlp import VocabBasedTokenizer

from .abstract_dataset_builder import AbstractDatasetBuilder


class TransformerTFRecordDatasetBuilder(AbstractDatasetBuilder):
    """Build dataset from tfrecord files."""

    def build_train_dataset(
            self,
            train_files,
            batch_size=32,
            src_max_len=512,
            tgt_max_len=512,
            record_option=None,
            skip_count=0,
            repeat_count=1,
            buffer_size=1000000,
            seed=None,
            reshuffle=True,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            dropout_remainder=False,
            **kwargs):

        dataset = tf.data.Dataset.from_tensor_slices(train_files)
        dataset = dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type=record_option).skip(skip_count),
            cycle_length=len(train_files))
        dataset = dataset.repeat(repeat_count)
        dataset = dataset.shuffle(
            buffer_size=buffer_size,
            seed=seed,
            reshuffle_each_iteration=reshuffle)

        def _parse_example_fn(x):
            name_to_features = {
                'src_ids': tf.io.FixedLenFeature([src_max_len], tf.int64),
                'tgt_ids': tf.io.FixedLenFeature([tgt_max_len], tf.int64)
            }
            example = tf.io.parse_single_example(x, name_to_features)
            features = example['src_ids']
            labels = example['tgt_ids']
            return (features, labels)

        dataset = dataset.map(lambda x: _parse_example_fn(x), num_parallel_calls=num_parallel_calls)
        dataset = dataset.filter(lambda x, y: tf.logical_and(tf.size(x) > 0, tf.size(y) > 0))
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=dropout_remainder)
        # should shift tgt sequence when generating tfrecord files
        dataset = dataset.map(lambda x, y: ((x, y[:, :-1]), y[:, 1:]), num_parallel_calls=num_parallel_calls)
        dataset = dataset.prefetch(batch_size)
        return dataset

    def build_valid_dataset(
            self,
            valid_files,
            batch_size=32,
            src_max_len=512,
            tgt_max_len=512,
            record_option=None,
            skip_count=0,
            repeat_count=1,
            buffer_size=1000000,
            seed=None,
            reshuffle=True,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            dropout_remainder=False,
            **kwargs):
        return self.build_train_dataset(
            valid_files,
            batch_size=batch_size,
            src_max_len=src_max_len,
            tgt_max_len=tgt_max_len,
            record_option=record_option,
            skip_count=skip_count,
            repeat_count=repeat_count,
            buffer_size=buffer_size,
            seed=seed,
            reshuffle=reshuffle,
            num_parallel_calls=num_parallel_calls,
            dropout_remainder=dropout_remainder,
            **kwargs)

    def build_predict_dataset(
            self,
            predict_files,
            batch_size=1,
            record_option=None,
            skip_count=0,
            repeat_count=1,
            src_max_len=512,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            **kwargs):
        dataset = tf.data.Dataset.from_tensor_slices(predict_files)
        dataset = dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type=record_option).skip(skip_count),
            cycle_length=len(predict_files))
        dataset = dataset.repeat(repeat_count)

        def _parse_example_fn(x):
            name_to_features = {
                'src_ids': tf.io.FixedLenFeature([src_max_len], tf.int64),
            }
            example = tf.io.parse_single_example(x, name_to_features)
            features = example['src_ids']
            return features

        dataset = dataset.map(lambda x: _parse_example_fn(x), num_parallel_calls=num_parallel_calls)
        dataset = dataset.filter(lambda x: tf.size(x) > 0)
        # x[0] is bos_id
        # ((x, [bos]), None)
        dataset = dataset.map(lambda x: ((x, [x[0]], None)))
        dataset = dataset.batch(batch_size).prefetch(batch_size)
        return dataset


class TransformerTextFileDatasetBuilder(AbstractDatasetBuilder):

    def __init__(self, src_tokenizer: VocabBasedTokenizer, tgt_tokenizer: VocabBasedTokenizer, **kwargs):
        super().__init__(**kwargs)
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenzier = tgt_tokenizer

    def build_train_dataset(
            self,
            train_files,
            src_max_len=None,
            tgt_max_len=None,
            num_bukets=1,
            batch_size=32,
            skip_count=0,
            repeat_count=1,
            buffer_size=1000000,
            seed=None,
            reshuffle=True,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            dropout_remainder=False,
            **kwargs):
        assert all(isinstance(f, tuple)
                   for f in train_files), "Each element of `train_files` must be an tuple of (src_file, tgt_file)"

        src_files, tgt_files = [x[0] for x in train_files], [x[1] for x in train_files]
        src_dataset = tf.data.TextLineDataset(src_files)
        tgt_dataset = tf.data.TextLineDataset(tgt_files)
        dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
        dataset = dataset.repeat(repeat_count)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=reshuffle)

        def _parse(x, y):
            src = x.numpy().decode('utf8')
            tgt = y.numpy().decode('utf8')
            src = self.src_tokenizer.encode(src, add_bos=True, add_eos=True)
            tgt = self.tgt_tokenzier.encode(tgt, add_bos=True, add_eos=True)
            return src, tgt

        def _parse_example(src, tgt):
            src_ids, tgt_ids = tf.py_function(_parse, [src, tgt], [tf.int64, tf.int64])
            return src_ids, tgt_ids

        # convert tokens to ids
        dataset = dataset.map(lambda x, y: _parse_example(x, y), num_parallel_calls=num_parallel_calls)

        # filter empty, >2 because add bos and eos ids to the sequence
        dataset = dataset.filter(lambda x, y: tf.logical_and(tf.size(x) > 2, tf.size(y) > 2))

        src_pad_id = tf.constant(self.src_tokenizer.pad_id, dtype=tf.int64)
        tgt_pad_id = tf.constant(self.tgt_tokenzier.pad_id, dtype=tf.int64)

        def batching_func(x):
            return x.padded_batch(
                batch_size=batch_size,
                padded_shapes=([src_max_len], [tgt_max_len]),
                padding_values=(src_pad_id, tgt_pad_id),
                drop_remainder=dropout_remainder)

        if num_bukets > 1:
            assert src_max_len is not None and src_max_len > 0, "src_max_len must be provid when num_buckets > 1"
            assert tgt_max_len is not None and tgt_max_len > 0, "tgt_max_len must be provid when num_buckets > 1"

            def key_func(x, y):
                if src_max_len:
                    bucket_width = (src_max_len + num_bukets - 1) // num_bukets
                else:
                    bucket_width = 10
                bucket_id = tf.maximum(src_max_len // bucket_width, tgt_max_len // bucket_width)
                return tf.cast(tf.minimum(num_bukets, bucket_id), tf.int64)

            def reduce_func(unsed_key, window_data):
                return batching_func(window_data)

            dataset = dataset.apply(
                tf.data.experimental.group_by_window(
                    key_func=key_func, reduce_func=reduce_func, window_size=batch_size))
        else:
            dataset = batching_func(dataset)

        dataset = dataset.map(lambda x, y: ((x, y[:, :-1]), y[:, 1:]))
        dataset = dataset.prefetch(batch_size)
        return dataset

    def build_valid_dataset(
            self,
            valid_files,
            src_max_len=None,
            tgt_max_len=None,
            num_bukets=1,
            batch_size=32,
            skip_count=0,
            repeat_count=1,
            buffer_size=1000000,
            seed=None,
            reshuffle=True,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            dropout_remainder=False,
            **kwargs):
        return self.build_train_dataset(
            valid_files,
            src_max_len=src_max_len,
            tgt_max_len=tgt_max_len,
            num_bukets=num_bukets,
            batch_size=batch_size,
            skip_count=skip_count,
            repeat_count=repeat_count,
            buffer_size=buffer_size,
            seed=seed,
            reshuffle=reshuffle,
            num_parallel_calls=num_bukets,
            dropout_remainder=dropout_remainder,
            **kwargs)

    def build_predict_dataset(
            self,
            predict_files,
            batch_size=1,
            repeat_count=1,
            src_max_len=None,
            skip_count=0,
            **kwargs):
        # dataset = tf.data.TextLineDataset(predict_files).skip(skip_count)
        # print('text line')
        # print(next(iter(dataset)))
        # dataset = dataset.repeat(repeat_count)
        # print('repeat:')
        # print(next(iter(dataset)))

        # def _parse_predict(s):
        #     text = s.numpy().decode('utf8')
        #     print(text)
        #     ids = self.src_tokenizer.encode(text, add_bos=True, add_eos=True)
        #     return ids

        # def _parse_predict_example(x):
        #     ids = tf.py_function(_parse_predict, [x], [tf.int64])
        #     return ids

        # dataset = dataset.map(lambda x: _parse_predict_example(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # print('After tokenize:')
        # print(next(iter(dataset)))
        # src_pad_id = tf.constant(self.src_tokenizer.pad_id, dtype=tf.int64)
        # dataset = dataset.batch(4)
        # print(next(iter(dataset)))
        # dataset = dataset.map(lambda x: ((x, ), None)).prefetch(batch_size)
        # print(next(iter(dataset)))
        # return dataset
        raise NotImplementedError()
