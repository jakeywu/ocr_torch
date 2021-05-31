import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(
            self,
            classes_num,
            rnn_type,
            hidden_size,
            num_layers,
            bidirectional,
            backbone,
    ):
        super(CRNN, self).__init__()

        self.backbone = self._call_backbone(backbone)
        self.rnn_in_channel = self.backbone.output_channel
        self.classes_num = classes_num
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        assert self.rnn_type in ["LSTM", "GRU"]

        if self.rnn_type == "LSTM":
            self.rnn_layer = nn.GRU(
                input_size=self.rnn_in_channel,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=False,
                bidirectional=bidirectional
            )
        else:
            self.rnn_layer = nn.LSTM(
                input_size=self.rnn_in_channel,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=False,
                bidirectional=self.bidirectional
            )
        for name, params in self.rnn_layer.named_parameters():
            nn.init.uniform_(params, -0.1, 0.1)

        rnn_out_channel = hidden_size
        if self.bidirectional:
            rnn_out_channel = hidden_size * 2

        self.fc = nn.Linear(
            in_features=rnn_out_channel,
            out_features=self.classes_num,
            bias=True
        )
        self.apply(self._weights_init)

    @staticmethod
    def _call_backbone(backbone):
        module_func = backbone.pop("name")
        if module_func == "rec_mobilenet_v3":
            from nets.rec.mobilenet_v3 import rec_mobilenet_v3
            module_func = eval(module_func)(**backbone)
        else:
            raise Exception("backbone {} is not found".format(module_func))
        return module_func

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    def forward(self, x):
        """
        :param x: N * 3 * H * W
        :return: # N * T * Feature
        """
        x = self.backbone(x)  # N * C * 1 * W (mobilenet: N * 288 * 1 * 25)
        x = x.squeeze(axis=2)  # N * C * W
        x = x.permute(2, 0, 1)  # N * W * C
        x, _ = self.rnn_layer(x)
        x = self.fc(x)
        if not self.training:
            x = F.softmax(x, dim=2)
        return x


if __name__ == "__main__":
    input_ = torch.randn(8, 3, 32, 320)
    se = CRNN(5000, "GRU", 48, 2, True, {"name": "rec_mobilenet_v3"})
    # se.eval()
    print(se(input_).shape)
