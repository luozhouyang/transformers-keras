"""Opinion Term Extraction and Classification"""

from transformers_keras.question_answering.models import AlbertForQuestionAnsweringX, BertForQuestionAnsweringX


class BertForOpinionTermExtractionAndClassification(BertForQuestionAnsweringX):
    """Use QA model to extract opinion term and classify sentiment polarity."""

    pass


class AlbertForOpinionTermExtractionAndClassification(AlbertForQuestionAnsweringX):
    """Use QA model to extract opinion term and classify sentiment polarity."""

    pass
