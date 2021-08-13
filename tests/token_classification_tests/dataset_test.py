import unittest

from transformers_keras.token_classification.dataset import TokenClassificationDataset
from transformers_keras.token_classification.parser import TokenClassificationExampleParserForChinese


class DatasetTest(unittest.TestCase):
    """Dataset tests."""

    def test_dataset(self):
        parser_fn = TokenClassificationExampleParserForChinese(
            vocab_file="testdata/vocab.bert.txt",
            label_vocab_file="testdata/labels.txt",
        )
        dataset = TokenClassificationDataset.from_conll_files(
            input_files="testdata/conll.txt",
            fn=parser_fn,
            sep="\\s+",
            batch_size=2,
        )
        print()
        print(next(iter(dataset)))


if __name__ == "__main__":
    unittest.main()
