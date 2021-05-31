import math
import torch
import torch.nn as nn
from torch.utils import model_zoo

__all__ = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


M_URLS = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicBlock(nn.Module):
    """
    通道可以升维降维, feature map可以上采样下采样
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None):
        super().__init__()

        # 当stride=2时，　下采样．
        # 当planes != inplanes * self.expansion时，升维
        self.conv1 = nn.Conv2d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=(3, 3),
            stride=stride,
            padding=(1, 1),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)  # inplace代表是否修改输入对象的值

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4  # 输出通道的倍乘

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=(3, 3),
            stride=stride,
            padding=(1, 1),
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes * self.expansion,
            kernel_size=(1, 1),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64  # 最开始默认输入维度为64
        self.layer_out_channels = [64, 128, 256, 512]
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.layer1 = self._make_layer(block, self.layer_out_channels[0], layers[0])
        self.layer2 = self._make_layer(block, self.layer_out_channels[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, self.layer_out_channels[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, self.layer_out_channels[3], layers[3], stride=(2, 2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=(1, 1)):
        downsample = None
        if stride != (1, 1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=(1, 1),
                    stride=stride,
                    padding=(0, 0),
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x2, x3, x4, x5


def resnet18(pre_trained_dir=None):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pre_trained_dir is None:
        return model

    state_dict = model_zoo.load_url(M_URLS["resnet18"], model_dir=pre_trained_dir)
    model.load_state_dict(state_dict, strict=False)
    return model


def resnet34(pre_trained_dir=None):
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pre_trained_dir is None:
        return model

    state_dict = model_zoo.load_url(M_URLS["resnet34"], model_dir=pre_trained_dir)
    model.load_state_dict(state_dict, strict=False)  # strict是否关注load_state_dict在恢复过程中的部分信息
    return model


def resnet50(pre_trained_dir=None):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pre_trained_dir is None:
        return model
    state_dict = model_zoo.load_url(M_URLS["resnet50"], model_dir=pre_trained_dir)
    model.load_state_dict(state_dict, strict=False)
    return model


def resnet101(pre_trained_dir=None):
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pre_trained_dir is None:
        return model

    state_dict = model_zoo.load_url(M_URLS["resnet101"], model_dir=pre_trained_dir)
    model.load_state_dict(state_dict, strict=False)
    return model


def resnet152(pre_trained_dir=None):
    model = ResNet(Bottleneck, [3, 8, 38, 3])
    if pre_trained_dir is None:
        return model

    state_dict = model_zoo.load_url(M_URLS["resnet101"], model_dir=pre_trained_dir)
    model.load_state_dict(state_dict, strict=False)
    return model


if __name__ == "__main__":
    data = torch.randn(8, 3, 640, 640)
    rn_model = resnet18("//pretrained_models/")
    fs = rn_model(data)
    for f in fs:
        print(f.shape)
