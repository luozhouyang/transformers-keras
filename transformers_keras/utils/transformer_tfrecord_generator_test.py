import unittest

import tensorflow as tf

from .transformer_tfrecord_generator import TransformerTFRecordGenerator


class TransformerTFRecordGeneratorTest(unittest.TestCase):

    def testGenerateTransformerTFRecordFile(self):
        src_vocab_file = 'testdata/vocab_src.txt'
        tgt_vocab_file = 'testdata/vocab_tgt.txt'
        config = {
            'src_unk_token': '<unk>',
            'src_sos_token': '<s>',
            'src_eos_token': '</s>',
            'src_pad_token': '<pad>',
            'tgt_unk_token': '<unk>',
            'tgt_sos_token': '<s>',
            'tgt_eos_token': '</s>',
            'tgt_pad_token': '<pad>',
            'record_option': 'GZIP',
            'max_src_sequence_length': 16,
            'max_tgt_sequence_length': 16
        }
        g = TransformerTFRecordGenerator(
            src_vocab_file=src_vocab_file, tgt_vocab_file=tgt_vocab_file, share_vocab=False, **config)
        input_files = [
            'testdata/train.txt'
        ]
        output_files = [
            'testdata/transformer_train.tfrecord'
        ]
        g.generate(input_files, output_files)

    def testReadTransformerTFRecordFile(self):
        f = 'testdata/transformer_train.tfrecord'
        dataset = tf.data.TFRecordDataset(f, compression_type='GZIP')
        dataset = dataset.repeat(10)
        for d in dataset.take(1):
            print(d)

        name_to_features = {
            'src_ids': tf.io.FixedLenFeature([16], tf.int64),
            'tgt_ids': tf.io.FixedLenFeature([16], tf.int64)
        }

        def _parse_fn(record):
            example = tf.io.parse_single_example(record, name_to_features)
            return example

        dataset = dataset.map(lambda x: _parse_fn(x))
        dataset = dataset.batch(2)
        for d in dataset.take(1):
            print(d)


if __name__ == "__main__":
    unittest.main()
