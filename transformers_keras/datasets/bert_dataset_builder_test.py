import tensorflow as tf

from .bert_dataset_builder import BertTFRecordDatasetBuilder


class BertDatasetBuilderTest(tf.test.TestCase):

    def testBertTFRecordDatasetBuilder(self):
        dataset_builder = BertTFRecordDatasetBuilder(
            record_option='GZIP',
            max_sequence_length=512,
        )

        files = ['testdata/bert_custom_pretrain.tfrecord']
        train_dataset = dataset_builder.build_train_dataset(files, batch_size=2, repeat_count=100)
        for d in train_dataset.take(2):
            print(d)


if __name__ == "__main__":
    tf.test.main()
