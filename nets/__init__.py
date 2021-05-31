from .det.dbnet import DBNet
from .rec.rnn import CRNN


__all__ = ["build_model"]


def build_model(config):
    module_name = config.pop("name")
    support_dict = ["DBNet", "CRNN"]
    assert module_name in support_dict
    module_class = eval(module_name)(**config)
    return module_class
