import math
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR, ExponentialLR, LambdaLR


class LearningSchedule(object):

    def __init__(
            self,
            optimizer,
            epochs,
            step_each_epoch,
            lr_method="_cosine_warmup",
            warmup_epoch=2,
            last_epoch=-1
    ):
        self.optimizer = optimizer
        self._last_epoch = step_each_epoch * last_epoch if last_epoch != -1 else -1
        self.t_max = step_each_epoch * epochs
        self.warmup_epoch = step_each_epoch * warmup_epoch
        assert lr_method in ["_step_lr", "_multi_step_lr", "_exponential_lr", "_cosine_annealing_lr",
                             "_cosine_warmup"]
        self._lr_method = lr_method

    @property
    def get_learning_rate(self):
        learning_rate = getattr(self, self._lr_method)()
        return learning_rate

    def _step_lr(self, gama=0.9, step_size=5):
        """
        固定步长衰减  lr  = base_lr * gama ** (epoch // step_size)
        :gama  学习率调整倍数
        :step_size 指的是Epoch间隔数
        """
        return StepLR(
            optimizer=self.optimizer,
            step_size=step_size,
            gamma=gama,
            last_epoch=self._last_epoch
        )

    def _multi_step_lr(self, milestones=None, gama=0.9):
        """
        多步长衰减
        :param milestones:　区间epoch
        :param gama: 学习率调整倍数
        """
        if not milestones:
            milestones = [10, 20, 50]
        return MultiStepLR(
            optimizer=self.optimizer,
            milestones=milestones,
            gamma=gama,
            last_epoch=self._last_epoch
        )

    def _exponential_lr(self, gama=0.98):
        """
        lr = base_lr * gama ** epoch
        指数衰减
        """
        return ExponentialLR(
            optimizer=self.optimizer,
            gamma=gama,
            last_epoch=self._last_epoch
        )

    def _cosine_annealing_lr(self, eta_min=0):
        """
        余弦退火衰减
        :t_max  最大迭代次数，　step_each_epoch * epoch
        """
        return CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.t_max,
            eta_min=eta_min,
            last_epoch=self._last_epoch,
        )

    def _cosine_warmup(self, eta_min=0):
        def lr_lambda(epoch):
            if epoch < self.warmup_epoch:
                return (epoch + 1) / self.warmup_epoch
            return eta_min + 0.5 * (
                        math.cos((epoch - self.warmup_epoch) / (self.t_max - self.warmup_epoch) * math.pi) + 1)

        return LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lr_lambda,
            last_epoch=self._last_epoch
        )


if __name__ == "__main__":
    from torchvision.models import resnet18
    from torch.optim import Adam

    model = resnet18(pretrained=False)
    _epochs = 10
    _step_each_epoch = 100
    optimizer1 = Adam(model.parameters(), lr=0.01)
    ls_schedule = LearningSchedule(
        optimizer=optimizer1,
        epochs=_epochs,
        step_each_epoch=_step_each_epoch,
        warmup_epoch=2
    ).get_learning_rate
    for _epoch in range(1, _epochs + 1):
        for _iter in range(1, _step_each_epoch + 1):
            lr = optimizer1.param_groups[0]["lr"]
            print("{}/{} {}/{} {:.8f}".format(_epoch, _epochs, _iter, _step_each_epoch, lr))
            optimizer1.zero_grad()
            optimizer1.step()
            ls_schedule.step()
