import abc
import json
import os
from collections import namedtuple


class AbstractBertClient(abc.ABC):
    """Abstract client"""

    def predict(self, input, **kwargs):
        raise NotImplementedError()

    def predict_batch(self, inputs, **kwargs):
        raise NotImplementedError()


class BertClientForMaskedLanguageModel(AbstractBertClient):
    """Bert client for MLM"""

    def predict(self, input, **kwargs):
        return super().predict(input, **kwargs)

    def predict_batch(self, inputs, **kwargs):
        return super().predict_batch(inputs, **kwargs)
