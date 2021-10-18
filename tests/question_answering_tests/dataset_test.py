import json
import unittest

from transformers_keras.datapipe.qa_dataset import DatasetForQuestionAnswering, DatasetForQuestionAnsweringX


class DatasetTest(unittest.TestCase):
    """Dataset test."""

    def _read_qa_instances(self, input_file):
        instances = []
        with open(input_file, mode="rt", encoding="utf-8") as fin:
            for line in fin:
                data = json.loads(line)
                instances.append(
                    {
                        "context": data["passage"],
                        "question": data["question"],
                        "answer": data["answer"],
                    }
                )
        return instances

    def _read_qax_instances(self, input_file):
        instances = []
        with open(input_file, mode="rt", encoding="utf-8") as fin:
            for line in fin:
                data = json.loads(line)
                instances.append(
                    {
                        "context": data["passage"],
                        "question": data["question"],
                        "answer": data["answer"],
                        "class": data["class"],
                    }
                )
        return instances

    def test_qa_dataset(self):
        instances = self._read_qa_instances("testdata/qa.sogouqa.jsonl")
        # test jsonl to examples
        examples = DatasetForQuestionAnswering.jsonl_to_examples(
            ["testdata/qa.sogouqa.jsonl"], vocab_file="testdata/vocab.bert.txt", context_key="passage"
        )
        for idx, e in enumerate(examples):
            print("{} -> {}".format(instances[idx]["answer"], " ".join(e.tokens[e.start : e.end + 1])))

        # test examples to tfrecord
        DatasetForQuestionAnswering.examples_to_tfrecord(examples, "testdata/qa.sogouqa.tfrecord")
        d = DatasetForQuestionAnswering.from_tfrecord_files(
            ["testdata/qa.sogouqa.tfrecord"] * 4,
            batch_size=4,
        )
        print()
        print(next(iter(d)))

        # test jsonl to tfrecord
        DatasetForQuestionAnswering.jsonl_to_tfrecord(
            "testdata/qa.sogouqa.jsonl",
            vocab_file="testdata/vocab.bert.txt",
            output_files="testdata/qa.sogouqa.tfrecord",
            context_key="passage",
        )

        # test from tfrecord
        d = DatasetForQuestionAnswering.from_tfrecord_files("testdata/qa.sogouqa.tfrecord", batch_size=4)
        print()
        print(next(iter(d)))

        # test from jsonl
        d = DatasetForQuestionAnswering.from_jsonl_files(
            "testdata/qa.sogouqa.jsonl",
            vocab_file="testdata/vocab.bert.txt",
            context_key="passage",
            batch_size=4,
        )
        print()
        print(next(iter(d)))

    def test_qax_dataset(self):
        instances = self._read_qax_instances("testdata/qax.sogouqa.jsonl")
        # test jsonl to examples
        examples = DatasetForQuestionAnsweringX.jsonl_to_examples(
            ["testdata/qax.sogouqa.jsonl"],
            vocab_file="testdata/vocab.bert.txt",
            context_key="passage",
        )
        for idx, e in enumerate(examples):
            print("{} -> {}".format(instances[idx]["answer"], " ".join(e.tokens[e.start : e.end + 1])))

        # test examples to tfrecord
        DatasetForQuestionAnsweringX.examples_to_tfrecord(examples, "testdata/qax.sogouqa.tfrecord")
        d = DatasetForQuestionAnsweringX.from_tfrecord_files(
            ["testdata/qax.sogouqa.tfrecord"] * 4,
            batch_size=4,
        )
        print()
        print(next(iter(d)))

        # test jsonl to tfrecord
        DatasetForQuestionAnsweringX.jsonl_to_tfrecord(
            "testdata/qax.sogouqa.jsonl",
            vocab_file="testdata/vocab.bert.txt",
            output_files="testdata/qax.sogouqa.tfrecord",
            context_key="passage",
        )

        # test from tfrecord
        d = DatasetForQuestionAnsweringX.from_tfrecord_files("testdata/qax.sogouqa.tfrecord", batch_size=4)
        print()
        print(next(iter(d)))

        # test from jsonl
        d = DatasetForQuestionAnsweringX.from_jsonl_files(
            "testdata/qax.sogouqa.jsonl", vocab_file="testdata/vocab.bert.txt", context_key="passage", batch_size=4
        )
        print()
        print(next(iter(d)))


if __name__ == "__main__":
    unittest.main()
