import json
import unittest

from tokenizers import BertWordPieceTokenizer
from transformers_keras.sequence_classification.dataset import (
    SequenceClassificationDataset,
    SequenceClassificationExample,
)


class DatasetTest(unittest.TestCase):
    """Dataset test."""

    def setUp(self) -> None:
        self.tokenizer = BertWordPieceTokenizer.from_file("testdata/vocab.bert.txt")

    def _read_examples(self, input_files):
        examples = []
        for f in input_files:
            with open(f, mode="rt", encoding="utf-8") as fin:
                for line in fin:
                    instance = json.loads(line)
                    encoding = self.tokenizer.encode(instance["sequence"])
                    examples.append(
                        SequenceClassificationExample(
                            tokens=encoding.tokens,
                            input_ids=encoding.ids,
                            segment_ids=encoding.type_ids,
                            attention_mask=encoding.attention_mask,
                            label=int(instance["label"]),
                        )
                    )
        return examples

    def test_sequence_classification_dataset(self):
        examples = self._read_examples(["testdata/sequence_classify.jsonl"])
        dataset = SequenceClassificationDataset.from_examples(
            examples,
            batch_size=4,
        )
        print()
        print(next(iter(dataset)))

        SequenceClassificationDataset.examples_to_tfrecord(examples, ["testdata/sequence_classify.tfrecord"])
        dataset = SequenceClassificationDataset.from_tfrecord_files(
            ["testdata/sequence_classify.tfrecord"] * 4,
            batch_size=4,
        )
        print()
        print(next(iter(dataset)))


if __name__ == "__main__":
    unittest.main()
