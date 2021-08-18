import json
import unittest

from tokenizers import BertWordPieceTokenizer
from transformers_keras.sentence_embedding.simcse_dataset import SimCSEDataset, SimCSEExample


class SimCSEDatasetTest(unittest.TestCase):
    """SimCSE dataset tests."""

    def setUp(self) -> None:
        self.tokenizer = BertWordPieceTokenizer.from_file("testdata/vocab.bert.txt")

    def _read_examples(self, input_files):
        examples = []
        for f in input_files:
            with open(f, mode="rt", encoding="Utf-8") as fin:
                for line in fin:
                    instance = json.loads(line)
                    anc_encoding = self.tokenizer.encode(instance["sequence"])
                    pos_encoding = self.tokenizer.encode(instance["pos_sequence"])
                    neg_encoding = self.tokenizer.encode(instance["neg_sequence"])
                    examples.append(
                        SimCSEExample(
                            sequence=instance["sequence"],
                            input_ids=anc_encoding.ids,
                            segment_ids=anc_encoding.type_ids,
                            attention_mask=anc_encoding.attention_mask,
                            pos_sequence=instance["pos_sequence"],
                            pos_input_ids=pos_encoding.ids,
                            pos_segment_ids=pos_encoding.type_ids,
                            pos_attention_mask=pos_encoding.attention_mask,
                            neg_sequence=instance["neg_sequence"],
                            neg_input_ids=neg_encoding.ids,
                            neg_segment_ids=neg_encoding.type_ids,
                            neg_attention_mask=neg_encoding.attention_mask,
                        )
                    )
        return examples

    def test_unsup_dataset_examples(self):
        examples = self._read_examples(["testdata/simcse.jsonl"])
        dataset = SimCSEDataset.from_examples(
            examples,
            with_pos_sequence=False,
            with_neg_sequence=False,
            batch_size=2,
        )
        print()
        print(next(iter(dataset)))

        SimCSEDataset.examples_to_tfrecord(
            examples,
            ["testdata/simcse.unsup.tfrecord"],
            with_pos_sequence=False,
            with_neg_sequence=False,
        )
        dataset = SimCSEDataset.from_tfrecord_files(
            ["testdata/simcse.unsup.tfrecord"] * 4,
            batch_size=2,
            with_pos_sequence=False,
            with_neg_sequence=False,
        )
        print()
        print(next(iter(dataset)))

    def test_unsup_dataset_jsonl(self):
        # test from jsonl
        dataset = SimCSEDataset.from_jsonl_files(
            "testdata/simcse.jsonl",
            "testdata/vocab.bert.txt",
            batch_size=2,
            with_pos_sequence=False,
            with_neg_sequence=False,
        )
        print()
        print(next(iter(dataset)))

        # test jsonl to tfrecord
        SimCSEDataset.jsonl_to_tfrecord(
            "testdata/simcse.jsonl",
            "testdata/vocab.bert.txt",
            ["testdata/simcse.unsup.tfrecord"],
            with_pos_sequence=False,
            with_neg_sequence=False,
        )
        dataset = SimCSEDataset.from_tfrecord_files(
            ["testdata/simcse.unsup.tfrecord"] * 4,
            batch_size=4,
            with_pos_sequence=False,
            with_neg_sequence=False,
        )
        print()
        print(next(iter(dataset)))

    def test_supervised_dataset_examples(self):
        examples = self._read_examples(["testdata/simcse.jsonl"])
        dataset = SimCSEDataset.from_examples(
            examples,
            with_pos_sequence=True,
            with_neg_sequence=False,
            batch_size=2,
        )
        print()
        print(next(iter(dataset)))

        SimCSEDataset.examples_to_tfrecord(
            examples,
            ["testdata/simcse.sup.tfrecord"],
            with_pos_sequence=True,
            with_neg_sequence=False,
        )
        dataset = SimCSEDataset.from_tfrecord_files(
            ["testdata/simcse.sup.tfrecord"] * 4,
            batch_size=4,
            with_pos_sequence=True,
            with_neg_sequence=False,
        )
        print()
        print(next(iter(dataset)))

    def test_supervised_dataset_jsonl(self):
        # test from jsonl
        dataset = SimCSEDataset.from_jsonl_files(
            "testdata/simcse.jsonl",
            "testdata/vocab.bert.txt",
            batch_size=2,
            with_pos_sequence=True,
            with_neg_sequence=False,
        )
        print()
        print(next(iter(dataset)))

        # test jsonl to tfrecord
        SimCSEDataset.jsonl_to_tfrecord(
            "testdata/simcse.jsonl",
            "testdata/vocab.bert.txt",
            ["testdata/simcse.sup.tfrecord"],
            with_pos_sequence=True,
            with_neg_sequence=False,
        )
        dataset = SimCSEDataset.from_tfrecord_files(
            ["testdata/simcse.sup.tfrecord"] * 4,
            batch_size=4,
            with_pos_sequence=True,
            with_neg_sequence=False,
        )
        print()
        print(next(iter(dataset)))

    def test_hardneg_dataset_examples(self):
        examples = self._read_examples(["testdata/simcse.jsonl"])
        dataset = SimCSEDataset.from_examples(
            examples,
            with_pos_sequence=False,
            with_neg_sequence=True,
            batch_size=2,
        )
        print()
        print(next(iter(dataset)))

        SimCSEDataset.examples_to_tfrecord(
            examples,
            ["testdata/simcse.hardneg.tfrecord"],
            with_pos_sequence=False,
            with_neg_sequence=True,
        )
        dataset = SimCSEDataset.from_tfrecord_files(
            ["testdata/simcse.hardneg.tfrecord"] * 4,
            batch_size=4,
            with_pos_sequence=False,
            with_neg_sequence=True,
        )
        print()
        print(next(iter(dataset)))

    def test_hardneg_dataset_jsonl(self):
        # test from jsonl
        dataset = SimCSEDataset.from_jsonl_files(
            "testdata/simcse.jsonl",
            "testdata/vocab.bert.txt",
            batch_size=2,
            with_pos_sequence=False,
            with_neg_sequence=True,
        )
        print()
        print(next(iter(dataset)))

        # test jsonl to tfrecord
        SimCSEDataset.jsonl_to_tfrecord(
            "testdata/simcse.jsonl",
            "testdata/vocab.bert.txt",
            ["testdata/simcse.hardneg.tfrecord"],
            with_pos_sequence=False,
            with_neg_sequence=True,
        )
        dataset = SimCSEDataset.from_tfrecord_files(
            ["testdata/simcse.hardneg.tfrecord"] * 4,
            batch_size=4,
            with_pos_sequence=False,
            with_neg_sequence=True,
        )
        print()
        print(next(iter(dataset)))


if __name__ == "__main__":
    unittest.main()
