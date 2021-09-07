import unittest

from transformers_keras.sentence_embedding.simcse_dataset import (
    HardNegativeSimCSEDataset,
    SupervisedSimCSEDataset,
    UnsupervisedSimCSEDataset,
)


class SimCSEDatasetTest(unittest.TestCase):
    """SimCSE dataset tests."""

    def test_unsup_dataset(self):
        print("====from_jsonl_files\n")
        d = UnsupervisedSimCSEDataset.from_jsonl_files(
            "testdata/simcse.jsonl", vocab_file="testdata/vocab.bert.txt", batch_size=2
        )
        print(next(iter(d)))

        print("====jsonl_to_examples\n")
        examples = UnsupervisedSimCSEDataset.jsonl_to_examples("testdata/simcse.jsonl", vocab_file="testdata/vocab.bert.txt")
        for i in range(2):
            print(examples[i])

        print("====from_examples\n")
        d = UnsupervisedSimCSEDataset.from_examples(examples, batch_size=2)
        print(next(iter(d)))

        print("====examples_to_tfrecord\n")
        UnsupervisedSimCSEDataset.examples_to_tfrecord(examples, "testdata/simcse.unsup.tfrecord")

        print("====from_tfrecord_files\n")
        d = UnsupervisedSimCSEDataset.from_tfrecord_files("testdata/simcse.unsup.tfrecord", batch_size=2)
        print(next(iter(d)))

    def test_supervised_dataset(self):
        print("====from_jsonl_files\n")
        d = SupervisedSimCSEDataset.from_jsonl_files(
            "testdata/simcse.jsonl", vocab_file="testdata/vocab.bert.txt", batch_size=2
        )
        print(next(iter(d)))

        print("====jsonl_to_examples\n")
        examples = SupervisedSimCSEDataset.jsonl_to_examples("testdata/simcse.jsonl", vocab_file="testdata/vocab.bert.txt")
        for i in range(2):
            print(examples[i])

        print("====from_examples\n")
        d = SupervisedSimCSEDataset.from_examples(examples, batch_size=2)
        print(next(iter(d)))

        print("====examples_to_tfrecord\n")
        SupervisedSimCSEDataset.examples_to_tfrecord(examples, "testdata/simcse.sup.tfrecord")

        print("====from_tfrecord_files\n")
        d = SupervisedSimCSEDataset.from_tfrecord_files("testdata/simcse.sup.tfrecord", batch_size=2)
        print(next(iter(d)))

    def test_hardneg_dataset(self):
        print("====from_jsonl_files\n")
        d = HardNegativeSimCSEDataset.from_jsonl_files(
            "testdata/simcse.jsonl", vocab_file="testdata/vocab.bert.txt", batch_size=2
        )
        print(next(iter(d)))

        print("====jsonl_to_examples\n")
        examples = HardNegativeSimCSEDataset.jsonl_to_examples("testdata/simcse.jsonl", vocab_file="testdata/vocab.bert.txt")
        for i in range(2):
            print(examples[i])

        print("====from_examples\n")
        d = HardNegativeSimCSEDataset.from_examples(examples, batch_size=2)
        print(next(iter(d)))

        print("====examples_to_tfrecord\n")
        HardNegativeSimCSEDataset.examples_to_tfrecord(examples, "testdata/simcse.hardneg.tfrecord")

        print("====from_tfrecord_files\n")
        d = HardNegativeSimCSEDataset.from_tfrecord_files("testdata/simcse.hardneg.tfrecord", batch_size=2)
        print(next(iter(d)))


if __name__ == "__main__":
    unittest.main()
