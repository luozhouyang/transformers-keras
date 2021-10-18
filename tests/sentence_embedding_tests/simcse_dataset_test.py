import unittest

from transformers_keras.datapipe.se_dataset import (
    DatasetForHardNegativeSimCSE,
    DatasetForSupervisedSimCSE,
    DatasetForUnsupervisedSimCSE,
)


class SimCSEDatasetTest(unittest.TestCase):
    """SimCSE dataset tests."""

    def test_unsup_dataset(self):
        print("====from_jsonl_files\n")
        d = DatasetForUnsupervisedSimCSE.from_jsonl_files(
            "testdata/simcse.jsonl", vocab_file="testdata/vocab.bert.txt", batch_size=2
        )
        print(next(iter(d)))

        print("====jsonl_to_examples\n")
        examples = DatasetForUnsupervisedSimCSE.jsonl_to_examples(
            "testdata/simcse.jsonl", vocab_file="testdata/vocab.bert.txt"
        )
        for i in range(2):
            print(examples[i])

        print("====from_examples\n")
        d = DatasetForUnsupervisedSimCSE.from_examples(examples, batch_size=2)
        print(next(iter(d)))

        print("====examples_to_tfrecord\n")
        DatasetForUnsupervisedSimCSE.examples_to_tfrecord(examples, "testdata/simcse.unsup.tfrecord")

        print("====from_tfrecord_files\n")
        d = DatasetForUnsupervisedSimCSE.from_tfrecord_files("testdata/simcse.unsup.tfrecord", batch_size=2)
        print(next(iter(d)))

    def test_supervised_dataset(self):
        print("====from_jsonl_files\n")
        d = DatasetForSupervisedSimCSE.from_jsonl_files(
            "testdata/simcse.jsonl", vocab_file="testdata/vocab.bert.txt", batch_size=2
        )
        print(next(iter(d)))

        print("====jsonl_to_examples\n")
        examples = DatasetForSupervisedSimCSE.jsonl_to_examples(
            "testdata/simcse.jsonl", vocab_file="testdata/vocab.bert.txt"
        )
        for i in range(2):
            print(examples[i])

        print("====from_examples\n")
        d = DatasetForSupervisedSimCSE.from_examples(examples, batch_size=2)
        print(next(iter(d)))

        print("====examples_to_tfrecord\n")
        DatasetForSupervisedSimCSE.examples_to_tfrecord(examples, "testdata/simcse.sup.tfrecord")

        print("====from_tfrecord_files\n")
        d = DatasetForSupervisedSimCSE.from_tfrecord_files("testdata/simcse.sup.tfrecord", batch_size=2)
        print(next(iter(d)))

    def test_hardneg_dataset(self):
        print("====from_jsonl_files\n")
        d = DatasetForHardNegativeSimCSE.from_jsonl_files(
            "testdata/simcse.jsonl", vocab_file="testdata/vocab.bert.txt", batch_size=2
        )
        print(next(iter(d)))

        print("====jsonl_to_examples\n")
        examples = DatasetForHardNegativeSimCSE.jsonl_to_examples(
            "testdata/simcse.jsonl", vocab_file="testdata/vocab.bert.txt"
        )
        for i in range(2):
            print(examples[i])

        print("====from_examples\n")
        d = DatasetForHardNegativeSimCSE.from_examples(examples, batch_size=2)
        print(next(iter(d)))

        print("====examples_to_tfrecord\n")
        DatasetForHardNegativeSimCSE.examples_to_tfrecord(examples, "testdata/simcse.hardneg.tfrecord")

        print("====from_tfrecord_files\n")
        d = DatasetForHardNegativeSimCSE.from_tfrecord_files("testdata/simcse.hardneg.tfrecord", batch_size=2)
        print(next(iter(d)))


if __name__ == "__main__":
    unittest.main()
