import unittest

from transformers_keras.question_answering.dataset import QuestionAnsweringDataset
from transformers_keras.question_answering.parser import QuestionAnsweringExampleParserForChinese


class DatasetTest(unittest.TestCase):
    """Dataset test."""

    def test_question_answering_dataset_for_chinese(self):
        parse_examples_fn = QuestionAnsweringExampleParserForChinese(
            vocab_file="testdata/vocab.bert.txt", context_key="passage"
        )
        d = QuestionAnsweringDataset.from_jsonl_files(
            input_files="testdata/qa.sogouqa.jsonl",
            fn=parse_examples_fn,
            batch_size=4,
        )
        print()
        print(next(iter(d)))


if __name__ == "__main__":
    unittest.main()
