import unittest

from transformers_keras.sentence_embedding.parser import SimCSEExampleParser
from transformers_keras.sentence_embedding.simcse_dataset import SimCSEDataset


class SimCSEDatasetTest(unittest.TestCase):
    """SimCSE dataset tests."""

    def test_unsup_dataset(self):
        parse_fn = SimCSEExampleParser(vocab_file="testdata/vocab.bert.txt")
        dataset = SimCSEDataset.from_jsonl_files(
            input_files="testdata/simcse.jsonl",
            fn=parse_fn,
            with_pos_sequence=False,
            with_neg_sequence=False,
            batch_size=2,
        )
        print()
        print(next(iter(dataset)))

    def test_supervised_dataset(self):
        parse_fn = SimCSEExampleParser(vocab_file="testdata/vocab.bert.txt")
        dataset = SimCSEDataset.from_jsonl_files(
            input_files="testdata/simcse.jsonl",
            fn=parse_fn,
            with_pos_sequence=True,
            with_neg_sequence=False,
            batch_size=2,
        )
        print()
        print(next(iter(dataset)))

    def test_hardneg_dataset(self):
        parse_fn = SimCSEExampleParser(vocab_file="testdata/vocab.bert.txt")
        dataset = SimCSEDataset.from_jsonl_files(
            input_files="testdata/simcse.jsonl",
            fn=parse_fn,
            with_pos_sequence=False,
            with_neg_sequence=True,
            batch_size=2,
        )
        print()
        print(next(iter(dataset)))


if __name__ == "__main__":
    unittest.main()
