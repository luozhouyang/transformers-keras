import tensorflow as tf
from naivenlp import TransformerTokenizer

from .transformer_dataset_builder import TransformerTextFileDatasetBuilder, TransformerTFRecordDatasetBuilder


class TransformerDatasetBuilderTest(tf.test.TestCase):

    def testTransformerTextFileDatasetBuilder(self):
        src_tokenizer = TransformerTokenizer(
            'testdata/vocab_src.txt',
            unk_token='<unk>', sos_token='<s>', eos_token='</s>', pad_token='<pad>')
        tgt_tokenizer = TransformerTokenizer(
            'testdata/vocab_tgt.txt',
            unk_token='<unk>', sos_token='<s>', eos_token='</s>', pad_token='<pad>')
        builder = TransformerTextFileDatasetBuilder(src_tokenizer, tgt_tokenizer)

        files = [('testdata/train.src.txt', 'testdata/train.tgt.txt')]

        train_dataset = builder.build_train_dataset(files, batch_size=4, repeat_count=100, buffer_size=100)
        for d in train_dataset.take(2):
            print(d)

        print('=' * 100)
        valid_dataset = builder.build_valid_dataset(files, batch_size=4, repeat_count=100, buffer_size=100)
        for d in valid_dataset.take(2):
            print(d)

    def testTransformerTFRecordDatasetBuilder(self):
        builder = TransformerTFRecordDatasetBuilder()

        files = ['testdata/transformer_train.tfrecord']
        train_dataset = builder.build_train_dataset(
            files, batch_size=4, repeat_count=100,
            src_max_len=16, tgt_max_len=16, record_option='GZIP', buffer_size=100)

        for d in train_dataset.take(2):
            print(d)

        print('=' * 100)
        valid_dataset = builder.build_valid_dataset(
            files, batch_size=4, repeat_count=100,
            src_max_len=16, tgt_max_len=16, record_option='GZIP', buffer_size=100)
        for d in valid_dataset.take(2):
            print(d)

        print('=' * 100)
        predict_dataset = builder.build_predict_dataset(
            ['testdata/transformer_train.tfrecord'], batch_size=4, repeat_count=100,
            src_max_len=16, tgt_max_len=16, record_option='GZIP', buffer_size=100)
        for d in predict_dataset.take(2):
            print(d)


if __name__ == "__main__":
    tf.test.main()
