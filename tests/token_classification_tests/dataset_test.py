import re
import unittest

from transformers_keras.token_classification.dataset import TokenClassificationDataset, TokenClassificationExample
from transformers_keras.token_classification.tokenizer import (
    TokenClassificationLabelTokenizer,
    TokenClassificationTokenizerForChinese,
)


class DatasetTest(unittest.TestCase):
    """Dataset tests."""

    def setUp(self) -> None:
        self.tokenizer = TokenClassificationTokenizerForChinese.from_file("testdata/vocab.bert.txt")
        self.label_tokenizer = TokenClassificationLabelTokenizer.from_file("testdata/labels.txt")

    def _read_examples(self, input_files, **kwargs):
        examples = []
        for f in input_files:
            features, labels = [], []
            with open(f, mode="rt", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        examples.append(self._compose_example(features, labels, **kwargs))
                        features, labels = [], []
                        continue
                    parts = re.split("\\s+", line)
                    if len(parts) != 2:
                        continue
                    features.append(parts[0])
                    labels.append(parts[1])
            if features and labels:
                examples.append(self._compose_example(features, labels, **kwargs))
        return examples

    def _compose_example(self, features, labels, **kwargs):
        ids = self.tokenizer.tokens_to_ids(features, add_cls=True, add_sep=True)
        label_ids = self.label_tokenizer.labels_to_ids(labels, add_cls=True, add_sep=True)
        return TokenClassificationExample(
            tokens=features,
            input_ids=ids,
            segment_ids=[0] * len(ids),
            attention_mask=[1] * len(ids),
            labels=labels,
            label_ids=label_ids,
        )

    def test_dataset(self):
        examples = self._read_examples(["testdata/conll.txt"])
        dataset = TokenClassificationDataset.from_examples(
            examples,
            batch_size=4,
        )
        print()
        print(next(iter(dataset)))

        TokenClassificationDataset.examples_to_tfrecord(examples, ["testdata/conll.tfrecord"])
        dataset = TokenClassificationDataset.from_tfrecord_files(
            ["testdata/conll.tfrecord"] * 4,
            batch_size=4,
        )
        print()
        print(next(iter(dataset)))


if __name__ == "__main__":
    unittest.main()
