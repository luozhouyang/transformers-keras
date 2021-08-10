import abc
import json
import re
from typing import List

from .dataset import QuestionAnsweringExample
from .tokenizer import QuestionAnsweringTokenizerForChinese

CHINESE_SEP_CHARACTERS = set(["，", "。", "：", "？", "！", "；", "’", ",", ":", "?", "!", ";"])


class AbstractQustionAnsweringExampleParser(abc.ABC):
    """Abstract example composer."""

    def __call__(self, *args, **kwargs) -> List[QuestionAnsweringExample]:
        """Parse examples for single instance(json line from input file)."""
        raise NotImplementedError()


class QuestionAnsweringExampleParserForChinese(AbstractQustionAnsweringExampleParser):
    """Question answering example composer for Chinese."""

    def __init__(
        self, vocab_file, question_key="question", context_key="context", answer_key="answer", sep_chars=None, **kwargs
    ) -> None:
        super().__init__()
        self.tokenizer = QuestionAnsweringTokenizerForChinese.from_file(vocab_file=vocab_file, **kwargs)
        self.question_key = question_key
        self.context_key = context_key
        self.answer_key = answer_key
        self.sep_chars = sep_chars or CHINESE_SEP_CHARACTERS

    def __call__(self, instance, max_sequence_length=512, **kwargs) -> List[QuestionAnsweringExample]:
        question = instance[self.question_key]
        question_encoding = self.tokenizer.encode(question, add_cls=True, add_sep=True)
        max_context_length = max_sequence_length - len(question) - 3
        context_examples = self._process_context_examples(
            instance[self.context_key], max_context_length=max_context_length, **kwargs
        )
        examples = []
        for context_example in context_examples:
            context, offset = context_example["context"], len(question_encoding.ids)
            context_encoding = self.tokenizer.encode(context, add_cls=False, add_sep=True)
            start, end = self._find_answer(instance[self.answer_key], context)
            if end > 0:
                assert str(context[start:end]).lower() == str(instance[self.answer_key]).lower()
            start += offset
            end += offset
            if end <= start:
                start, end = 0, 0
            examples.append(
                QuestionAnsweringExample(
                    text="[CLS]" + question + "[SEP]" + context + "[SEP]",
                    tokens=question_encoding.tokens + context_encoding.tokens,
                    input_ids=question_encoding.ids + context_encoding.ids,
                    segment_ids=[0] * len(question_encoding.ids) + [1] * len(context_encoding.ids),
                    attention_mask=[1] * (len(question_encoding.ids) + len(context_encoding.ids)),
                    start=start,
                    end=max(end - 1, 0),  # avoid end < 0
                )
            )
        return examples

    def _find_answer(self, answer, context):
        for m in re.finditer(re.escape(answer), context, re.IGNORECASE):
            start, end = m.span()
            return start, end
        return 0, 0

    def _process_context_examples(self, context, max_context_length, **kwargs):
        context_examples = []
        offset = 0
        for context in self._truncate_context(context, max_context_length=max_context_length, **kwargs):
            context_examples.append({"context": context, "offset": offset})
            offset += len(context)
        return context_examples

    def _truncate_context(self, context, max_context_length, **kwargs):
        truncated_sequences = []
        prev = 0
        while prev < len(context):
            idx = prev + max_context_length
            if idx >= len(context):
                truncated_sequences.append(context[prev:idx])
                break
            # context[idx] is not an seperator, find prev index of seperator
            left = self._find_prev_sep(context, idx, prev)
            # left seperator is less equal than prev, cut context directly
            if left <= prev:
                truncated_sequences.append(context[prev:idx])
                prev = idx
                continue
            # normal case, find seperator after prev
            truncated_sequences.append(context[prev : left + 1])
            prev = left + 1
        return truncated_sequences

    def _find_prev_sep(self, sequence, idx, prev):
        while idx >= 0 and sequence[idx] not in self.sep_chars:
            idx -= 1
        return idx
