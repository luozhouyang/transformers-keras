import abc
from re import M
from typing import List

from .dataset import TokenClassificationExample
from .tokenizer import TokenClassificationLabelTokenizer, TokenClassificationTokenizerForChinese


class AbstractTokenClassificationExampleParser(abc.ABC):
    """Abstract example parser for token classification."""

    @abc.abstractmethod
    def __call__(self, feature, label, **kwargs) -> List[TokenClassificationExample]:
        raise NotImplementedError()


class TokenClassificationExampleParserForChinese(AbstractTokenClassificationExampleParser):
    """Example parser for chinese."""

    def __init__(self, vocab_file, label_vocab_file, do_lower_case=True, o_token="O", **kwargs) -> None:
        super().__init__()
        self.tokenizer = TokenClassificationTokenizerForChinese.from_file(
            vocab_file, do_lower_case=do_lower_case, **kwargs
        )
        self.label_tokenizer = TokenClassificationLabelTokenizer.from_file(label_vocab_file, o_token=o_token, **kwargs)

    def __call__(self, features, labels, **kwargs) -> List[TokenClassificationExample]:
        examples = []
        input_ids = self.tokenizer.tokens_to_ids(features, add_cls=True, add_sep=True)
        label_ids = self.label_tokenizer.labels_to_ids(labels, add_cls=True, add_sep=True)
        examples.append(
            TokenClassificationExample(
                tokens=features,
                labels=labels,
                input_ids=input_ids,
                segment_ids=[0] * len(input_ids),
                attention_mask=[1] * len(input_ids),
                label_ids=label_ids,
            )
        )
        return examples
