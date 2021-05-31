from .det_postprocess import DBPostProcess
from .rec_postprocess import CRnnPostProcess


__all__ = ["build_post_process"]


def build_post_process(post_config):
    module_name = post_config.pop("name")
    assert module_name in ["DBPostProcess", "CRnnPostProcess"]
    module_class = eval(module_name)(**post_config)
    return module_class
