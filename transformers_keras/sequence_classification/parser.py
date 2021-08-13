import abc
import json
import logging
from typing import List

from tokenizers import BertWordPieceTokenizer

from .dataset import SequenceClassificationExample


class AbstractSequenceClassificationExampleParser(abc.ABC):
    """Abstract sequence classification example parser."""

    @abc.abstractmethod
    def __call__(self, instance, **kwargs) -> List[SequenceClassificationExample]:
        raise NotImplementedError()


class SequenceClassificationExampleParser(AbstractSequenceClassificationExampleParser):
    """Sequence classification example parser."""

    def __init__(self, vocab_file, sequence_key="sequence", label_key="label", do_lower_case=True, **kwargs) -> None:
        super().__init__()
        self.sequence_key = sequence_key
        self.label_key = label_key
        self.tokenizer = BertWordPieceTokenizer.from_file(vocab_file, lowercase=do_lower_case, **kwargs)

    def __call__(self, instance, **kwargs) -> List[SequenceClassificationExample]:
        examples = []
        encoding = self.tokenizer.encode(instance[self.sequence_key])
        examples.append(
            SequenceClassificationExample(
                tokens=encoding.tokens,
                input_ids=encoding.ids,
                segment_ids=encoding.type_ids,
                attention_mask=encoding.attention_mask,
                label=int(instance[self.label_key]),
            )
        )
        return examples
