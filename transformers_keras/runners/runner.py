import abc


class Runner(abc.ABC):

    def train(self):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    def export(self):
        raise NotImplementedError()
