from .det_dataset import DetDataSet
from .rec_dataset import RecDataSet
from torch.utils.data import DistributedSampler, DataLoader


__all__ = ["build_data_loader"]


def build_data_loader(config, distributed, logger, mode):
    module_name = config[mode]["dataset"]["name"]
    data_set = eval(module_name)(config, logger, mode)
    sampler = None
    if distributed:
        sampler = DistributedSampler(data_set)

    loader_conf = config[mode]["dataloader"]
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=loader_conf["batch_size"],
        shuffle=(sampler is None),  # when distributed is True, shuffle is False
        drop_last=loader_conf["drop_last"],
        num_workers=loader_conf["num_workers"],
        sampler=sampler,
        pin_memory=loader_conf["pin_memory"],  # 提高数据转移速度
    )
    return data_loader
