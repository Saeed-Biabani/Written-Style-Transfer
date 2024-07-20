from torch import nn

class Discriminator(nn.Sequential):
  def __init__(self, in_channels):
    super(Discriminator, self).__init__(
        ResDownLayer(in_channels, 32),
        ResDownLayer(32, 64),
        ResDownLayer(64, 128),
        ResDownLayer(128, 128),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, 1),
    )


class ResDownLayer(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ResDownLayer, self).__init__()
    self.__block = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
        nn.AvgPool2d(2)
    )
    self.residual = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1),
        nn.AvgPool2d(2)
    )
  def forward(self, x):
    res = self.__block(x)
    return self.residual(x) + res