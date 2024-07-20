from einops.layers.torch import Rearrange
from torchvision.models import vgg16
from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = VGGEncoder()
        self.pool = nn.MaxPool2d(2)
        self.reshape = Rearrange("b c h w -> b (h w) c")
    
    def forward(self, x):
        skip = []
        for layer in self.layers:
            x = layer(x)
            skip.append(x)
            x = self.pool(x)
        x = self.reshape(x)
        return x, skip


def VGGEncoder():
    base_model = vgg16()
    base_model.training = True
        
    encoder_seq =  nn.ModuleList()
    moduls = nn.Sequential()
    for layer in list(base_model.features.children()):
        if isinstance(layer, nn.modules.pooling.MaxPool2d):
            encoder_seq.append(moduls)
            moduls = nn.Sequential()
        else:
            moduls.append(layer)
    encoder_seq[0][0] = nn.Conv2d(1, 64, 3, 1, 1)
    return encoder_seq