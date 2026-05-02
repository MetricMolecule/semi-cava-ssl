
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
