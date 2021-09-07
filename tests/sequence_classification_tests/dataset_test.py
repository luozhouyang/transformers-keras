import unittest

from transformers_keras.sequence_classification.dataset import SequenceClassificationDataset


class DatasetTest(unittest.TestCase):
    """Dataset test."""

    def test_sequence_classification_dataset_examples(self):
        print()
        print("====from_jsonl_files")
        d = SequenceClassificationDataset.from_jsonl_files(
            "testdata/sequence_classify.jsonl", vocab_file="testdata/vocab.bert.txt", batch_size=2
        )
        print(next(iter(d)))

        print("====jsonl_to_examples")
        examples = SequenceClassificationDataset.jsonl_to_examples(
            "testdata/sequence_classify.jsonl", vocab_file="testdata/vocab.bert.txt"
        )
        for i in range(2):
            print(examples[i])

        print("====from_examples")
        d = SequenceClassificationDataset.from_examples(examples, batch_size=2)
        print(next(iter(d)))

        print("====examples_to_tfrecord")
        SequenceClassificationDataset.examples_to_tfrecord(examples, output_files=["testdata/sequence_classify.tfrecord"])

        print("====from_tfrecord_files")
        d = SequenceClassificationDataset.from_tfrecord_files("testdata/sequence_classify.tfrecord", batch_size=2)
        print(next(iter(d)))


if __name__ == "__main__":
    unittest.main()
