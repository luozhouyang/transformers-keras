import unittest

from transformers_keras.datapipe.sa_dataset import DatasetForAspectTermExtraction


class DatasetTest(unittest.TestCase):
    """Dataset test"""

    def test_build_dataset(self):
        print("====from_jsonl_files\n")
        d = DatasetForAspectTermExtraction.from_jsonl_files(
            "testdata/ate.jsonl", vocab_file="testdata/vocab.bert.txt", batch_size=2
        )
        print(next(iter(d)))

        print("====jsonl_to_examples\n")
        examples = DatasetForAspectTermExtraction.jsonl_to_examples(
            "testdata/ate.jsonl", vocab_file="testdata/vocab.bert.txt"
        )
        for i in range(2):
            print(examples[i])

        print("====from_examples\n")
        d = DatasetForAspectTermExtraction.from_examples(examples, batch_size=2)
        print(next(iter(d)))

        print("====examples_to_tfrecord\n")
        DatasetForAspectTermExtraction.examples_to_tfrecord(examples, output_files=["testdata/ate.tfrecord"])

        print("====from_tfrecord_files\n")
        d = DatasetForAspectTermExtraction.from_tfrecord_files(input_files=["testdata/ate.tfrecord"])
        print(next(iter(d)))


if __name__ == "__main__":
    unittest.main()
