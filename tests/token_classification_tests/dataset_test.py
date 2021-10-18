import json
import unittest

from transformers_keras.datapipe.tc_dataset import DatasetForTokenClassification


class DatasetTest(unittest.TestCase):
    """Dataset tests."""

    def generate_jsonl(self):
        examples = self._read_examples(["testdata/conll.txt"])
        with open("testdata/token_classify.jsonl", mode="wt", encoding="utf-8") as fout:
            for e in examples:
                info = {"features": e.tokens, "labels": e.labels}
                fout.write(json.dumps(info, ensure_ascii=False))
                fout.write("\n")

    def test_dataset(self):
        print("====from_jsonl_files")
        d = DatasetForTokenClassification.from_jsonl_files(
            "testdata/token_classify.jsonl",
            vocab_file="testdata/vocab.bert.txt",
            label_vocab_file="testdata/labels.txt",
            batch_size=2,
        )
        print(next(iter(d)))

        print("====jsonl_to_examples")
        examples = DatasetForTokenClassification.jsonl_to_examples(
            "testdata/token_classify.jsonl",
            vocab_file="testdata/vocab.bert.txt",
            label_vocab_file="testdata/labels.txt",
        )
        for i in range(2):
            print(examples[i])

        print("====from_examples")
        d = DatasetForTokenClassification.from_examples(examples, batch_size=2)
        print(next(iter(d)))

        print("====examples_to_tfrecord")
        DatasetForTokenClassification.examples_to_tfrecord(examples, ["testdata/token_classify.tfrecord"])

        print("====from_tfrecord_files")
        d = DatasetForTokenClassification.from_tfrecord_files("testdata/token_classify.tfrecord")
        print(next(iter(d)))


if __name__ == "__main__":
    unittest.main()
