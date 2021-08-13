import abc
from typing import List

from tokenizers import BertWordPieceTokenizer

from .simcse_dataset import SimCSEExample


class AbstractSimCSEExampleParser:
    """Abstract simcse example parser."""

    @abc.abstractmethod
    def __call__(self, instance, **kwargs):
        raise NotImplementedError()


class SimCSEExampleParser:
    """SimCSE example parser"""

    def __init__(self, vocab_file, do_lower_case=True, **kwargs) -> None:
        self.tokenizer = BertWordPieceTokenizer.from_file(vocab_file, lowercase=do_lower_case)
        self.sequence_key = kwargs.pop("sequence_key", "sequence")
        self.pos_sequence_key = kwargs.pop("pos_sequence_key", "pos_sequence")
        self.neg_sequence_key = kwargs.pop("neg_sequence_key", "neg_sequence")

    def __call__(self, instance, with_pos_sequence=False, with_neg_sequence=False, **kwargs) -> List[SimCSEExample]:
        examples = []
        sequence_encoding = self.tokenizer.encode(instance[self.sequence_key])
        if with_neg_sequence:
            pos_sequence_encoding = self.tokenizer.encode(instance[self.pos_sequence_key])
            neg_sequence_encoding = self.tokenizer.encode(instance[self.neg_sequence_key])
            examples.append(
                SimCSEExample(
                    sequence=instance[self.sequence_key],
                    input_ids=sequence_encoding.ids,
                    segment_ids=sequence_encoding.type_ids,
                    attention_mask=sequence_encoding.attention_mask,
                    pos_sequence=instance[self.pos_sequence_key],
                    pos_input_ids=pos_sequence_encoding.ids,
                    pos_segment_ids=pos_sequence_encoding.type_ids,
                    pos_attention_mask=pos_sequence_encoding.attention_mask,
                    neg_sequence=instance[self.neg_sequence_key],
                    neg_input_ids=neg_sequence_encoding.ids,
                    neg_segment_ids=neg_sequence_encoding.type_ids,
                    neg_attention_mask=neg_sequence_encoding.attention_mask,
                )
            )
            return examples
        if with_pos_sequence:
            pos_sequence_encoding = self.tokenizer.encode(instance[self.pos_sequence_key])
            examples.append(
                SimCSEExample(
                    sequence=instance[self.sequence_key],
                    input_ids=sequence_encoding.ids,
                    segment_ids=sequence_encoding.type_ids,
                    attention_mask=sequence_encoding.attention_mask,
                    pos_sequence=instance[self.pos_sequence_key],
                    pos_input_ids=pos_sequence_encoding.ids,
                    pos_segment_ids=pos_sequence_encoding.type_ids,
                    pos_attention_mask=pos_sequence_encoding.attention_mask,
                    neg_sequence=None,
                    neg_input_ids=None,
                    neg_segment_ids=None,
                    neg_attention_mask=None,
                )
            )
            return examples

        examples.append(
            SimCSEExample(
                sequence=instance[self.sequence_key],
                input_ids=sequence_encoding.ids,
                segment_ids=sequence_encoding.type_ids,
                attention_mask=sequence_encoding.attention_mask,
                pos_sequence=None,
                pos_input_ids=None,
                pos_segment_ids=None,
                pos_attention_mask=None,
                neg_sequence=None,
                neg_input_ids=None,
                neg_segment_ids=None,
                neg_attention_mask=None,
            )
        )
        return examples
