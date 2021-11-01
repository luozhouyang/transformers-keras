import unittest

from transformers_keras.datapipe.sc_dataset import DataPipeForSequenceClassification


class DatasetTest(unittest.TestCase):
    """Dataset test."""

    def test_sequence_classification_dataset_examples(self):
        print()
        print("====from_jsonl_files")
        d = DataPipeForSequenceClassification.from_jsonl_files(
            "testdata/sequence_classify.jsonl", vocab_file="testdata/vocab.bert.txt", batch_size=2
        )
        print(next(iter(d)))

        print("====jsonl_to_examples")
        examples = DataPipeForSequenceClassification.jsonl_to_examples(
            "testdata/sequence_classify.jsonl", vocab_file="testdata/vocab.bert.txt"
        )
        for i in range(2):
            print(examples[i])

        print("====from_examples")
        d = DataPipeForSequenceClassification.from_examples(examples, batch_size=2)
        print(next(iter(d)))

        print("====examples_to_tfrecord")
        DataPipeForSequenceClassification.examples_to_tfrecord(
            examples, output_files=["testdata/sequence_classify.tfrecord"]
        )

        print("====from_tfrecord_files")
        d = DataPipeForSequenceClassification.from_tfrecord_files("testdata/sequence_classify.tfrecord", batch_size=2)
        print(next(iter(d)))


if __name__ == "__main__":
    unittest.main()
