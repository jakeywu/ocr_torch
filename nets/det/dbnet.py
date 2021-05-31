import torch
import torch.nn as nn


class DBNet(nn.Module):

    def __init__(
            self,
            inner_channel,
            k,
            backbone
    ):
        """
        :param inner_channel: FPN对齐维度
        :param k:
        :param backbone:
        """
        super(DBNet, self).__init__()
        self.inner_channel = inner_channel
        self.k = k
        self.backbone = self._call_backbone(backbone)
        self.channel_size_lst = self.backbone.layer_out_channels
        self.in5 = nn.Conv2d(
            in_channels=self.channel_size_lst[-1],
            out_channels=self.inner_channel,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False
        )
        self.in4 = nn.Conv2d(
            in_channels=self.channel_size_lst[-2],
            out_channels=self.inner_channel,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False
        )
        self.in3 = nn.Conv2d(
            in_channels=self.channel_size_lst[-3],
            out_channels=self.inner_channel,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False
        )
        self.in2 = nn.Conv2d(
            in_channels=self.channel_size_lst[-4],
            out_channels=self.inner_channel,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False
        )

        self.up5 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")

        self.out5 = nn.Sequential(
            nn.Conv2d(self.inner_channel, self.inner_channel//4, (3, 3), (1, 1), (1, 1), bias=False),
            nn.Upsample(scale_factor=8, mode="nearest")
        )
        self.out4 = nn.Sequential(
            nn.Conv2d(self.inner_channel, self.inner_channel//4, (3, 3), (1, 1), (1, 1), bias=False),
            nn.Upsample(scale_factor=4, mode="nearest")
        )
        self.out3 = nn.Sequential(
            nn.Conv2d(self.inner_channel, self.inner_channel//4, (3, 3), (1, 1), (1, 1), bias=False),
            nn.Upsample(scale_factor=2, mode="nearest")
        )
        self.out2 = nn.Conv2d(self.inner_channel, self.inner_channel//4, (3, 3), (1, 1), (1, 1), bias=False)

        self.binarize = nn.Sequential(
            nn.Conv2d(self.inner_channel, self.inner_channel//4, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(self.inner_channel//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.inner_channel//4, self.inner_channel//4, (2, 2), (2, 2)),
            nn.BatchNorm2d(self.inner_channel//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.inner_channel//4, 1, (2, 2), (2, 2)),
            nn.Sigmoid()
        )

        self.thresh = nn.Sequential(
            nn.Conv2d(self.inner_channel, self.inner_channel//4, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(self.inner_channel//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.inner_channel//4, self.inner_channel//4, (2, 2), (2, 2)),
            nn.BatchNorm2d(self.inner_channel//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.inner_channel//4, 1, (2, 2), (2, 2)),
            nn.Sigmoid()
        )
        self.weights_init()

    @staticmethod
    def _call_backbone(backbone):
        module_func = backbone.pop("name")
        if module_func == "det_mobilenet_v3":
            from nets.det.mobilenetv3 import det_mobilenet_v3
            module_func = eval(module_func)(**backbone)
        else:
            from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
            module_func = eval(module_func)(**backbone)
        return module_func

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)

    def step_function(self, x, y):
        return torch.reciprocal((1+torch.exp(-self.k * (x-y))))

    def forward(self, x):
        """
        c2, c3, c4, c5依次为原图像H*W的1/4, 1/8, 1/16, 1/32
        假设H*W为640*640
        """
        features = self.backbone(x)
        c2, c3, c4, c5 = features

        # Channel统一
        in5 = self.in5(c5)  # N * 256  * 20 * 20
        in4 = self.in4(c4)  # N * 256  * 40 * 40
        in3 = self.in3(c3)  # N * 256  * 80 * 80
        in2 = self.in2(c2)  # N * 256  * 160 * 160

        # 上层特征上采样,依次和下层特征合并
        out5 = in5  # N * 256  * 20 * 20
        out4 = self.up5(in5) + in4  # N * 256  * 40 * 40
        out3 = self.up4(in4) + in3  # N * 256  * 80 * 80
        out2 = self.up3(in3) + in2  # N * 256  * 160 * 160

        # 降维，特征上采样
        p5 = self.out5(out5)  # N * 64  * 160 * 160
        p4 = self.out4(out4)  # N * 64  * 160 * 160
        p3 = self.out3(out3)  # N * 64  * 160 * 160
        p2 = self.out2(out2)  # N * 64  * 160 * 160

        # 后Concat
        fuse = torch.cat((p5, p4, p3, p2), dim=1)  # # N * 256  * 160 * 160
        prob = self.binarize(fuse)
        if not self.training:
            # N * 1 * H * W
            return prob

        thresh = self.thresh(fuse)
        binary_thresh = self.step_function(prob, thresh)
        # N * 3 * H * W
        return torch.cat([prob, thresh, binary_thresh], dim=1)


if __name__ == "__main__":
    model = DBNet(96, 50, backbone={"name": "det_mobilenet_v3"})
    ignored_params = list(map(id, model.binarize.parameters()))
    print(ignored_params)
    # base_params = filter(lambda p: id(p) not in ignored_params,
    #                      model.parameters())
    #
    # optimizer = torch.optim.SGD([
    #     {'params': base_params},
    #     {'params': model.fc.parameters(), 'lr': 1e-3}
    # ], lr=1e-2, momentum=0.9)
    # print(model)
    # for param in model.parameters():
    #     print(param)
    #     import pdb
    #     pdb.set_trace()