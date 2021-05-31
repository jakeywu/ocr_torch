from torch import optim


class OptimizerScheduler(object):

    def __init__(self, parameters, init_learning_rate=0.001, optim_method="_adam"):
        self._parameters = parameters
        self._learning_rate = init_learning_rate
        self._method = optim_method
        self.optim = self.__getattribute__(self._method)()

    def _adam(self):
        return optim.Adam(
            params=self._parameters,
            lr=self._learning_rate,
            weight_decay=0,
            amsgrad=True
        )

    def _sgd(self):
        return optim.SGD(
            params=self._parameters,
            lr=self._learning_rate
        )
