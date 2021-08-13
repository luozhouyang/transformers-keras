import unittest

from transformers_keras.sequence_classification.dataset import SequenceClassificationDataset
from transformers_keras.sequence_classification.parser import SequenceClassificationExampleParser


class DatasetTest(unittest.TestCase):
    """Dataset test."""

    def test_sequence_classification_dataset(self):
        dataset = SequenceClassificationDataset.from_jsonl_files(
            input_files="testdata/sequence_classify.jsonl",
            fn=SequenceClassificationExampleParser(vocab_file="testdata/vocab.bert.txt"),
            batch_size=2,
        )
        print()
        print(next(iter(dataset)))


if __name__ == "__main__":
    unittest.main()
