import tensorflow as tf

from transformers_keras.tokenizers import TransformerDefaultTokenizer

from .transformer_dataset_builder import (TransformerTextFileDatasetBuilder,
                                          TransformerTFRecordDatasetBuilder)


class TransformerDatasetBuilderTest(tf.test.TestCase):

    def testTransformerTextFileDatasetBuilder(self):
        src_tokenizer = TransformerDefaultTokenizer(
            'testdata/vocab_src.txt',
            unk_token='<unk>', sos_token='<s>', eos_token='</s>', pad_token='<pad>')
        tgt_tokenizer = TransformerDefaultTokenizer(
            'testdata/vocab_tgt.txt',
            unk_token='<unk>', sos_token='<s>', eos_token='</s>', pad_token='<pad>')
        dataset_builder = TransformerTextFileDatasetBuilder(
            src_tokenizer, tgt_tokenizer,
            train_shuffle_buffer_size=100,
            valid_shuffle_buffer_size=100,
        )
        files = [('testdata/train.src.txt', 'testdata/train.tgt.txt')]

        train_dataset = dataset_builder.build_train_dataset(files)
        for d in train_dataset.take(2):
            print(d)

        valid_dataset = dataset_builder.build_valid_dataset(files)
        for d in valid_dataset.take(2):
            print(d)

    def testTransformerTFRecordDatasetBuilder(self):
        dataset_builder = TransformerTFRecordDatasetBuilder(
            src_max_len=16, tgt_max_len=16,
            train_shuffle_buffer_size=100,
            valid_shuffle_buffer_size=100,
        )

        files = ['testdata/transformer_train.tfrecord']

        train_dataset = dataset_builder.build_train_dataset(files)
        for d in train_dataset.take(2):
            print(d)

        valid_dataset = dataset_builder.build_valid_dataset(files)
        for d in valid_dataset.take(2):
            print(d)


if __name__ == "__main__":
    tf.test.main()
