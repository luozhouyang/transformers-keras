import json
import random
import unittest

from transformers_keras.question_answering.dataset import (
    QuestionAnsweringDataset, QuestionAnsweringExample,
    QuestionAnsweringXDataset, QuestionAnsweringXExample)
from transformers_keras.question_answering.tokenizer import \
    QuestionAnsweringTokenizerForChinese


class DatasetTest(unittest.TestCase):
    """Dataset test."""

    def setUp(self) -> None:
        self.tokenizer = QuestionAnsweringTokenizerForChinese.from_file("testdata/vocab.bert.txt")

    def _read_qa_examples(self, input_files):
        examples = []
        for f in input_files:
            with open(f, mode="rt", encoding="utf-8") as fin:
                for line in fin:
                    instance = json.loads(line)
                    ctx_encoding = self.tokenizer.encode(instance["passage"], add_cls=True, add_sep=True)
                    qst_encoding = self.tokenizer.encode(instance["question"], add_cls=False, add_sep=True)
                    start, end = random.randint(0, 50), random.randint(50, 100)
                    examples.append(
                        QuestionAnsweringExample(
                            text="[CLS]" + instance["passage"] + "[SEP]" + instance["question"] + "[SEP]",
                            tokens=ctx_encoding.tokens + qst_encoding.tokens,
                            input_ids=ctx_encoding.ids + qst_encoding.ids,
                            segment_ids=[0] * len(ctx_encoding.ids) + [1] * len(qst_encoding.ids),
                            attention_mask=[1] * (len(ctx_encoding.ids) + len(qst_encoding.ids)),
                            start=start,
                            end=end,
                        )
                    )
        return examples

    def _read_qax_examples(self, input_files):
        examples = []
        for f in input_files:
            with open(f, mode="rt", encoding="utf-8") as fin:
                for line in fin:
                    instance = json.loads(line)
                    ctx_encoding = self.tokenizer.encode(instance["passage"], add_cls=True, add_sep=True)
                    qst_encoding = self.tokenizer.encode(instance["question"], add_cls=False, add_sep=True)
                    start, end = random.randint(0, 50), random.randint(50, 100)
                    examples.append(
                        QuestionAnsweringXExample(
                            text="[CLS]" + instance["passage"] + "[SEP]" + instance["question"] + "[SEP]",
                            tokens=ctx_encoding.tokens + qst_encoding.tokens,
                            input_ids=ctx_encoding.ids + qst_encoding.ids,
                            segment_ids=[0] * len(ctx_encoding.ids) + [1] * len(qst_encoding.ids),
                            attention_mask=[1] * (len(ctx_encoding.ids) + len(qst_encoding.ids)),
                            start=start,
                            end=end,
                            class_id=random.randint(0, 5),
                        )
                    )
        return examples

    def test_question_answering_dataset_for_chinese(self):
        examples = self._read_qa_examples(["testdata/qa.sogouqa.jsonl"])
        d = QuestionAnsweringDataset.from_examples(
            examples,
            batch_size=4,
        )
        print()
        print(next(iter(d)))

        QuestionAnsweringDataset.examples_to_tfrecord(examples, ["testdata/qa.sogouqa.tfrecord"])
        d = QuestionAnsweringDataset.from_tfrecord_files(
            ["testdata/qa.sogouqa.tfrecord"] * 4,
            batch_size=4,
        )
        print()
        print(next(iter(d)))

    def test_question_answering_datasetx(self):
        examples = self._read_qax_examples(["testdata/qax.sogouqa.jsonl"])
        d = QuestionAnsweringXDataset.from_examples(
            examples,
            batch_size=4,
        )
        print()
        print(next(iter(d)))

        QuestionAnsweringXDataset.examples_to_tfrecord(examples, ["testdata/qax.sogouqa.tfrecord"])
        d = QuestionAnsweringXDataset.from_tfrecord_files(
            ["testdata/qax.sogouqa.tfrecord"] * 4,
            batch_size=4,
        )
        print()
        print(next(iter(d)))


if __name__ == "__main__":
    unittest.main()
