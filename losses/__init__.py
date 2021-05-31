from .det_loss import L1BalanceCELoss
from .ctc_loss import CTCLoss


__all__ = ["build_loss"]


def build_loss(config):
    module_name = config.pop("name")
    support_dict = ["L1BalanceCELoss", "CTCLoss"]
    assert module_name in support_dict
    module_class = eval(module_name)(**config)
    return module_class
