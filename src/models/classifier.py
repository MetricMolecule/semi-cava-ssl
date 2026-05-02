
class Classifier(nn.Module):
    def __init__(self, zc_dim, K, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zc_dim, hid),
            nn.ReLU(True),
            nn.Linear(hid, K)
        )

    def forward(self, zc):
        return self.net(zc)
