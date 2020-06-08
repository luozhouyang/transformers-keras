import tensorflow as tf

from transformers_keras import (TransformerDefaultTokenizer, TransformerRunner,
                                TransformerTextFileDatasetBuilder)
from transformers_keras import BertTFRecordDatasetBuilder
from transformers_keras import BertRunner
from transformers_keras import AlbertRunner


class RunnersTest(tf.test.TestCase):

    def testTransformer(self):

        src_tokenizer = TransformerDefaultTokenizer(vocab_file='testdata/vocab_src.txt')
        tgt_tokenizer = TransformerDefaultTokenizer(vocab_file='testdata/vocab_tgt.txt')
        dataset_builder = TransformerTextFileDatasetBuilder(src_tokenizer, tgt_tokenizer, train_repeat_count=100)

        model_config = {
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'src_vocab_size': src_tokenizer.vocab_size,
            'tgt_vocab_size': tgt_tokenizer.vocab_size,
        }

        runner = TransformerRunner(model_config, dataset_builder, model_dir='models/transformer')

        train_files = [('testdata/train.src.txt', 'testdata/train.tgt.txt')]
        runner.train(train_files, epochs=10, callbacks=None)

    def testBert(self):
        dataset_builder = BertTFRecordDatasetBuilder(
            max_sequence_length=128, record_option='GZIP', train_repeat_count=100, eos_token='T')

        model_config = {
            'max_positions': 128,
            'num_layers': 6,
            'vocab_size': 21128,
        }

        runner = BertRunner(model_config, dataset_builder, model_dir='models/bert')

        train_files = ['testdata/bert_custom_pretrain.tfrecord']
        runner.train(train_files, epochs=10, callbacks=None)

    def testAlbert(self):
        # ALBERT has the same data format with BERT
        dataset_builder = BertTFRecordDatasetBuilder(
            max_sequence_length=128, record_option='GZIP', train_repeat_count=100, eos_token='T')

        model_config = {
            'max_positions': 128,
            'num_layers': 6,
            'num_groups': 1,
            'num_layers_each_group': 1,
            'vocab_size': 21128, 
        }

        runner = AlbertRunner(model_config, dataset_builder, model_dir='models/albert')

        train_files = ['testdata/bert_custom_pretrain.tfrecord']
        runner.train(train_files, epochs=10, callbacks=None)


if __name__ == "__main__":
    tf.test.main()
