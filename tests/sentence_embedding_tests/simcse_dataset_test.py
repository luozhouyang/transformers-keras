import unittest

from transformers_keras.datapipe.se_dataset import (
    DataPipeForHardNegativeSimCSE,
    DataPipeForSupervisedSimCSE,
    DataPipeForUnsupervisedSimCSE,
)


class SimCSEDatasetTest(unittest.TestCase):
    """SimCSE dataset tests."""

    def test_unsup_dataset(self):
        print("====from_jsonl_files\n")
        d = DataPipeForUnsupervisedSimCSE.from_jsonl_files(
            "testdata/simcse.jsonl", vocab_file="testdata/vocab.bert.txt", batch_size=2
        )
        print(next(iter(d)))

        print("====jsonl_to_examples\n")
        examples = DataPipeForUnsupervisedSimCSE.jsonl_to_examples(
            "testdata/simcse.jsonl", vocab_file="testdata/vocab.bert.txt"
        )
        for i in range(2):
            print(examples[i])

        print("====from_examples\n")
        d = DataPipeForUnsupervisedSimCSE.from_examples(examples, batch_size=2)
        print(next(iter(d)))

        print("====examples_to_tfrecord\n")
        DataPipeForUnsupervisedSimCSE.examples_to_tfrecord(examples, "testdata/simcse.unsup.tfrecord")

        print("====from_tfrecord_files\n")
        d = DataPipeForUnsupervisedSimCSE.from_tfrecord_files("testdata/simcse.unsup.tfrecord", batch_size=2)
        print(next(iter(d)))

    def test_supervised_dataset(self):
        print("====from_jsonl_files\n")
        d = DataPipeForSupervisedSimCSE.from_jsonl_files(
            "testdata/simcse.jsonl", vocab_file="testdata/vocab.bert.txt", batch_size=2
        )
        print(next(iter(d)))

        print("====jsonl_to_examples\n")
        examples = DataPipeForSupervisedSimCSE.jsonl_to_examples(
            "testdata/simcse.jsonl", vocab_file="testdata/vocab.bert.txt"
        )
        for i in range(2):
            print(examples[i])

        print("====from_examples\n")
        d = DataPipeForSupervisedSimCSE.from_examples(examples, batch_size=2)
        print(next(iter(d)))

        print("====examples_to_tfrecord\n")
        DataPipeForSupervisedSimCSE.examples_to_tfrecord(examples, "testdata/simcse.sup.tfrecord")

        print("====from_tfrecord_files\n")
        d = DataPipeForSupervisedSimCSE.from_tfrecord_files("testdata/simcse.sup.tfrecord", batch_size=2)
        print(next(iter(d)))

    def test_hardneg_dataset(self):
        print("====from_jsonl_files\n")
        d = DataPipeForHardNegativeSimCSE.from_jsonl_files(
            "testdata/simcse.jsonl", vocab_file="testdata/vocab.bert.txt", batch_size=2
        )
        print(next(iter(d)))

        print("====jsonl_to_examples\n")
        examples = DataPipeForHardNegativeSimCSE.jsonl_to_examples(
            "testdata/simcse.jsonl", vocab_file="testdata/vocab.bert.txt"
        )
        for i in range(2):
            print(examples[i])

        print("====from_examples\n")
        d = DataPipeForHardNegativeSimCSE.from_examples(examples, batch_size=2)
        print(next(iter(d)))

        print("====examples_to_tfrecord\n")
        DataPipeForHardNegativeSimCSE.examples_to_tfrecord(examples, "testdata/simcse.hardneg.tfrecord")

        print("====from_tfrecord_files\n")
        d = DataPipeForHardNegativeSimCSE.from_tfrecord_files("testdata/simcse.hardneg.tfrecord", batch_size=2)
        print(next(iter(d)))


if __name__ == "__main__":
    unittest.main()
