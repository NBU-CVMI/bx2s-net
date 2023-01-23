import torch.nn as nn


class SEch(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SEch, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).reshape(b, c)
        y = self.fc(y).reshape(b, c, 1, 1, 1)
        return x * y
