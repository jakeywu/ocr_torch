import torch
import torch.nn as nn
import torch.nn.functional as F
M_URL = "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


def _make_divisible(v, divisor=8, min_value=None):
    # 等比例的增加和减少通道的个数
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeBlock(nn.Module):
    def __init__(self, exp_size, divide=4):
        super(SqueezeBlock, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(exp_size, exp_size // divide),  # Squeeze线性连接
            nn.ReLU(inplace=True),
            nn.Linear(exp_size // divide, exp_size),  # Excite线性连接
        )

    def forward(self, x):
        batch, channels, height, width = x.size()
        # 1.池化
        out = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        out = self.dense(out)
        out = F.hardsigmoid(out, inplace=True)
        # resize
        out = out.view(batch, channels, 1, 1)
        # 相乘
        return out * x


class MobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, non_linear, _se, exp_size, dropout_rate=1.0):
        super(MobileBlock, self).__init__()
        self.out_channels = out_channels
        self.nonLinear = non_linear
        self.SE = _se
        self.dropout_rate = dropout_rate
        padding = (kernal_size - 1) // 2

        self.use_connect = (stride == 1 and in_channels == out_channels)  # 残差条件

        if self.nonLinear == "RE":
            activation = nn.ReLU
        else:
            activation = nn.Hardswish

        # 1*1卷积 expand
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, exp_size, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(exp_size),
            activation(inplace=True)
        )
        # 膨胀的卷积操作, 深度卷积 3* 3 或者 5*5
        self.depth_conv = nn.Sequential(
            nn.Conv2d(exp_size, exp_size, kernel_size=kernal_size, stride=stride, padding=padding, groups=exp_size),
            nn.BatchNorm2d(exp_size),
            activation(inplace=True)
        )

        if self.SE:
            self.squeeze_block = SqueezeBlock(exp_size)

        # 1*1卷积  逐点卷积
        self.point_conv = nn.Sequential(
            nn.Conv2d(exp_size, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Identity()
        )

    def forward(self, x):
        # MobileNetV2
        out = self.expand_conv(x)  # 1*1 卷积, 由输入通道转为膨胀通道，转换通道 in->exp
        out = self.depth_conv(out)  # 3x3或5*5卷积，膨胀通道，使用步长stride

        # Squeeze and Excite
        if self.SE:
            out = self.squeeze_block(out)

        # 1x1卷积，由膨胀通道，转换为输出通道
        out = self.point_conv(out)  # 转换通道 exp->out

        # 残差结构
        if self.use_connect:
            return x + out
        else:
            return out


class MobileNetV3(nn.Module):
    def __init__(self, multiplier=0.5, use_se=False):
        super(MobileNetV3, self).__init__()
        self._multiplier = multiplier
        self._use_se = use_se
        # in_channel  out_channel kernel_size stride nl se exp_size
        cfg = [
            # in_channel, out_channel, kernel_size,  stride, activation, se, exp_size
            [16, 16, 3, (1, 1), 'RE', True, 16],
            [16, 24, 3, (2, 1), 'RE', False, 72],
            [24, 24, 3, 1, 'RE', False, 88],
            [24, 40, 5, (2, 1), 'HS', True, 96],
            [40, 40, 5, 1, 'HS', True, 240],
            [40, 40, 5, 1, 'HS', True, 240],
            [40, 48, 5, 1, 'HS', True, 120],
            [48, 48, 5, 1, 'HS', True, 144],
            [48, 96, 5, (2, 1), 'HS', True, 288],
            [96, 96, 5, 1, 'HS', True, 576],
            [96, 96, 5, 1, 'HS', True, 576],
        ]
        cls_ch_squeeze = _make_divisible(576 * self._multiplier)
        self.output_channel = cls_ch_squeeze
        self.init_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=_make_divisible(16 * self._multiplier),
                kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(_make_divisible(16 * self._multiplier)),
            nn.Hardswish(inplace=True)
        )

        self.layer = self._make_layer(self._multi_layer_conf(cfg))
        self.last_conv = nn.Sequential(
            nn.Conv2d(_make_divisible(cfg[-1][1]), cls_ch_squeeze, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(cls_ch_squeeze),
            nn.Hardswish(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.apply(_weights_init)

    def _multi_layer_conf(self, layer_conf):
        for lc in layer_conf:
            lc[0] = _make_divisible(lc[0] * self._multiplier)
            lc[1] = _make_divisible(lc[1] * self._multiplier)
            lc[-1] = _make_divisible(lc[-1] * self._multiplier)
        return layer_conf

    def _make_layer(self, layer_conf, sequential=True):
        block_list = []
        for in_channels, out_channels, kernel_size, stride, activation, se, exp_size in layer_conf:
            # activation　 NL: 非线性激活函数；HS: H-swish激活函数，RE:ReLU激活函数
            # SE Squeeze and Excite结构，是否压缩和激发
            # exp_size: 膨胀参数
            # MobileBlock瓶颈层
            se = se and self._use_se
            block_list.append(MobileBlock(in_channels, out_channels, kernel_size, stride, activation, se, exp_size))
        if sequential:
            return nn.Sequential(*block_list)
        return block_list

    def forward(self, x):
        """
        :param x: N * C * H * W
        :return: N * 288 * 1 * 25
        """
        # 起始部分
        out = self.init_conv(x)
        # 中间部分
        out = self.layer(out)
        out = self.last_conv(out)
        out = self.pool(out)
        return out


def rec_mobilenet_v3(pre_trained_dir=None, multiplier=0.5, use_se=False):
    if not pre_trained_dir:
        return MobileNetV3(multiplier, use_se)


if __name__ == "__main__":
    model11 = rec_mobilenet_v3()
    import torchvision
    torchvision.models.mobilenet_v3_small()
    x1 = torch.randn(8, 3, 32, 320)
    fx = model11(x1)
    print(fx.shape)
