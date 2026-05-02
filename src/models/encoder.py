import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)

        self.dropout = nn.Dropout(p=drop_rate) if drop_rate > 0 else None

        self.shortcut = (
            nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)
            if in_planes != out_planes or stride != 1
            else None
        )

    def forward(self, x):
        out = self.relu(self.bn1(x))
        shortcut = x if self.shortcut is None else self.shortcut(out)

        out = self.conv1(out)
        out = self.relu(self.bn2(out))

        if self.dropout is not None:
            out = self.dropout(out)

        out = self.conv2(out)
        return out + shortcut


class NetBlock(nn.Module):
    def __init__(self, num_layers, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(
                BasicBlock(
                    in_planes if i == 0 else out_planes,
                    out_planes,
                    stride if i == 0 else 1,
                    drop_rate
                )
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self, depth=28, width=2, in_ch=3, drop=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        s = [16, 16 * width, 32 * width, 64 * width]

        self.conv = nn.Conv2d(in_ch, s[0], 3, 1, 1, bias=False)
        self.block1 = NetBlock(n, s[0], s[1], 1, drop)
        self.block2 = NetBlock(n, s[1], s[2], 2, drop)
        self.block3 = NetBlock(n, s[2], s[3], 2, drop)
        self.bn = nn.BatchNorm2d(s[3])
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.out_dim = s[3]

    def forward(self, x):
        x = self.conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.relu(self.bn(x), inplace=True)
        return self.pool(x).flatten(1)
