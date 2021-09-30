import unittest

from transformers_keras.lm.mlm_dataset import MaskedLanguageModelDataset


class MaskedLanguageModelDatasetTest(unittest.TestCase):
    """Test dataset for masked language model"""

    def test_mlm_dataset(self):
        print("\n=====jsonl_to_examples")
        examples = MaskedLanguageModelDataset.jsonl_to_examples(
            input_files=["testdata/mlm.jsonl"],
            vocab_file="testdata/vocab.bert.txt",
        )
        for idx, e in enumerate(examples):
            print("{} -> {}".format(idx, e))

        print("=====from_examples")
        d = MaskedLanguageModelDataset.from_examples(examples, batch_size=2)
        print(next(iter(d)))

        print("=====examples_to_tfrecord")
        MaskedLanguageModelDataset.examples_to_tfrecord(examples, output_files=["testdata/mlm.tfrecord"])
        print("=====from_tfrecord_files")
        d = MaskedLanguageModelDataset.from_tfrecord_files(input_files=["testdata/mlm.tfrecord"], batch_size=2)
        print(next(iter(d)))


if __name__ == "__main__":
    unittest.main()
