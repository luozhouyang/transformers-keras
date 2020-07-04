import abc
import os


class AbstractAdapter(abc.ABC):

    @abc.abstractmethod
    def adapte(self, pretrain_model_dir, checkpoint, **kwargs):
        raise NotImplementedError()
