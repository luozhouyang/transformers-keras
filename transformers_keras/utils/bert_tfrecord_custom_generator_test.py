import unittest

import tensorflow as tf

from .bert_tfrecord_custom_generator import CustomBertGenerator


class CustomBertGeneratorTest(unittest.TestCase):

    def testGenerateTFRecordFiles(self):
        config = {
            'do_lower_case': True,
            'split_on_punc': True,
            'max_sequence_length': 128,
            'max_predictions_per_seq': 20,
            'do_whole_word_mask': True,
            'masked_lm_prob': 0.15,
            'unk_token': '[UNK]',
            'pad_token': '[PAD]',
            'record_option': 'GZIP'
        }
        vocab_file = 'testdata/bert_vocab.txt'
        input_files = [
            'testdata/bert_custom_corpus.txt'
        ]
        output_files = [
            'testdata/bert_custom_pretrain.tfrecord'
        ]
        g = CustomBertGenerator(vocab_file, **config)
        g.generate(input_files, output_files)

    def testReadTFRecordFile(self):
        f = 'testdata/bert_custom_pretrain.tfrecord'
        dataset = tf.data.TFRecordDataset(f, compression_type='GZIP')
        for d in dataset.take(1):
            print(d)

        name_to_features = {
            'original_ids': tf.io.FixedLenFeature([128], tf.int64),
            'input_ids': tf.io.FixedLenFeature([128], tf.int64),
            'input_mask': tf.io.FixedLenFeature([128], tf.int64),
            'segment_ids': tf.io.FixedLenFeature([128], tf.int64),
            'next_sentence_labels': tf.io.FixedLenFeature([1], tf.int64),
            'masked_lm_positions': tf.io.FixedLenFeature([20], tf.int64),
            'masked_lm_weights': tf.io.FixedLenFeature([20], tf.float32),
            'masked_lm_ids': tf.io.FixedLenFeature([20], tf.int64),
        }

        def _parse_example(record):
            example = tf.io.parse_single_example(record, name_to_features)
            return example

        dataset = dataset.map(lambda x: _parse_example(x))
        dataset = dataset.batch(4)
        for d in dataset.take(1):
            print(d)


if __name__ == "__main__":
    unittest.main()
