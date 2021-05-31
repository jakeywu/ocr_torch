from .det_metric import DetMetric
from .rec_metric import RecMetric


__all__ = ["build_metric"]


def build_metric(config):
    module_name = config.pop("name")
    support_dict = ["DetMetric", "RecMetric"]
    assert module_name in support_dict
    module_class = eval(module_name)(**config)
    return module_class
