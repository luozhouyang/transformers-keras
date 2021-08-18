import json
import re
import unittest

from tokenizers import BertWordPieceTokenizer
from transformers_keras.token_classification.dataset import TokenClassificationDataset, TokenClassificationExample
from transformers_keras.token_classification.tokenizer import TokenClassificationLabelTokenizer


class DatasetTest(unittest.TestCase):
    """Dataset tests."""

    def setUp(self) -> None:
        self.tokenizer = BertWordPieceTokenizer.from_file("testdata/vocab.bert.txt")
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
        ids = [101] + [self.tokenizer.token_to_id(x) for x in features] + [102]
        label_ids = self.label_tokenizer.labels_to_ids(labels, add_cls=True, add_sep=True)
        return TokenClassificationExample(
            tokens=features,
            input_ids=ids,
            segment_ids=[0] * len(ids),
            attention_mask=[1] * len(ids),
            labels=labels,
            label_ids=label_ids,
        )

    def test_generate_jsonl(self):
        examples = self._read_examples(["testdata/conll.txt"])
        with open("testdata/token_classify.jsonl", mode="wt", encoding="utf-8") as fout:
            for e in examples:
                info = {"features": e.tokens, "labels": e.labels}
                fout.write(json.dumps(info, ensure_ascii=False))
                fout.write("\n")

    def test_dataset(self):
        # test from examples
        examples = self._read_examples(["testdata/conll.txt"])
        dataset = TokenClassificationDataset.from_examples(
            examples,
            batch_size=4,
        )
        print()
        print(next(iter(dataset)))

        # test examples to tfrecord
        TokenClassificationDataset.examples_to_tfrecord(examples, ["testdata/conll.tfrecord"])

        # test conll to tfrecord
        TokenClassificationDataset.conll_to_tfrecord(
            ["testdata/conll.txt"], "testdata/vocab.bert.txt", "testdata/labels.txt", ["testdata/conll.tfrecord"]
        )

        # test jsonl to tfrecord
        TokenClassificationDataset.jsonl_to_tfrecord(
            ["testdata/token_classify.jsonl"],
            "testdata/vocab.bert.txt",
            "testdata/labels.txt",
            ["testdata/token_classify.tfrecord"],
        )

        # test from conll files
        dataset = TokenClassificationDataset.from_conll_files(
            ["testdata/conll.txt"], "testdata/vocab.bert.txt", "testdata/labels.txt", batch_size=4
        )
        print()
        print(next(iter(dataset)))

        # test from jsonl files
        dataset = TokenClassificationDataset.from_jsonl_files(
            ["testdata/token_classify.jsonl"],
            "testdata/vocab.bert.txt",
            "testdata/labels.txt",
            batch_size=4,
        )
        print()
        print(next(iter(dataset)))

        # test from tfrecord
        dataset = TokenClassificationDataset.from_tfrecord_files(
            ["testdata/conll.tfrecord", "testdata/token_classify.tfrecord"],
            batch_size=4,
        )
        print()
        print(next(iter(dataset)))


if __name__ == "__main__":
    unittest.main()
