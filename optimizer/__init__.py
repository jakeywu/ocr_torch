from .optim import OptimizerScheduler
from .learning_rate import LearningSchedule


__all__ = ["build_optimizer"]


def build_optimizer(parameters, epochs, step_each_epoch, config):
    optimizer_name = config.pop("name")
    support_dict = ["OptimizerScheduler"]
    assert optimizer_name in support_dict
    optimizer = eval(optimizer_name)(
        parameters=parameters,
        optim_method=config["optim_method"],
        init_learning_rate=config["init_learning_rate"],
    ).optim

    lr_schedule_conf = config["learning_schedule"]
    lr_schedule_name = lr_schedule_conf["name"]
    lr_schedule = eval(lr_schedule_name)(
        optimizer=optimizer,
        epochs=epochs,
        step_each_epoch=step_each_epoch,
        warmup_epoch=lr_schedule_conf["warmup_epoch"],
        lr_method=lr_schedule_conf["lr_method"],
        last_epoch=-1
    ).get_learning_rate
    return optimizer, lr_schedule
