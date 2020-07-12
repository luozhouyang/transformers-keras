import unittest

import tensorflow as tf
from naivenlp import TransformerTokenizer

from .transformer_tfrecord_generator import TransformerTFRecordGenerator


class TransformerTFRecordGeneratorTest(unittest.TestCase):

    def testGenerateTransformerTFRecordFile(self):
        src_vocab_file = 'testdata/vocab_src.txt'
        tgt_vocab_file = 'testdata/vocab_tgt.txt'
        src_tokenizer = TransformerTokenizer(
            vocab_file=src_vocab_file,
            do_lower_case=True,
            do_basic_tokenization=True,
            tokenize_chinese_chars=True,
            never_split=None,
            pad_token='<pad>', unk_token='<unk>', bos_token='<s>', eos_token='</s>',
        )
        self.assertEqual(0, src_tokenizer.pad_id)
        self.assertEqual(1, src_tokenizer.unk_id)
        self.assertEqual(2, src_tokenizer.bos_id)
        self.assertEqual(3, src_tokenizer.eos_id)

        tgt_tokenizer = TransformerTokenizer(
            vocab_file=tgt_vocab_file,
            do_basic_tokenization=True,
            do_lower_case=True,
            tokenize_chinese_chars=True,
            nerver_split=None,
            pad_token='<pad>', unk_token='<unk>', bos_token='<s>', eos_token='</s>',
        )
        self.assertEqual(0, tgt_tokenizer.pad_id)
        self.assertEqual(1, tgt_tokenizer.unk_id)
        self.assertEqual(2, tgt_tokenizer.bos_id)
        self.assertEqual(3, tgt_tokenizer.eos_id)

        g = TransformerTFRecordGenerator(
            src_tokenizer,
            tgt_tokenizer,
            src_max_len=16,
            tgt_max_len=16,
            record_option='GZIP',
        )
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
