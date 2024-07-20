from einops.layers.torch import Rearrange
from torch.nn import functional as nnf
from torch import nn

class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.reshape = Rearrange("b (h w) c -> b c h w", h = cfg.min_h, w = cfg.min_w)
        
        self.layers = nn.ModuleList()
        self.layers.append(DecoderLayer(512, 512))
        self.layers.append(DecoderLayer(512, 256))
        self.layers.append(DecoderLayer(256, 128))
        self.layers.append(DecoderLayer(128, 64))
        self.layers.append(DecoderLayer(64, 32))
    
    def forward(self, x, connections):
        x = self.reshape(x)
        for layer, skip in zip(self.layers, reversed(connections)):
            x = layer(x, skip)
        
        return x
    

class ResUnit(nn.Module):
    def __init__(
        self,
        features,
        apply_bn = True,
        activation = nn.ReLU()
    ):
        super(ResUnit, self).__init__()
        self.__conv = nn.Sequential()

        self.__conv.append(activation)
        self.__conv.append(nn.Conv2d(features, features, 3, 1, 1, bias = not apply_bn))
        if apply_bn:
            self.__conv.append(nn.BatchNorm2d(features))

        self.__conv.append(activation)
        self.__conv.append(nn.Conv2d(features, features, 3, 1, 1, bias = not apply_bn))
        if apply_bn:
            self.__conv.append(nn.BatchNorm2d(features))
        
    def forward(self, x):
        out = self.__conv(x) + x
        return out

class DecoderLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
    ):
        super(DecoderLayer, self).__init__()

        self.res_1 = ResUnit(in_features)
        self.res_2 = ResUnit(in_features)
        self.cout = nn.Conv2d(in_features, out_features, 1, 1, 0, bias = True)

    def forward(self, x, skip):
        skip_out = self.res_1(skip)

        out = self.res_2(x)
        out = nnf.interpolate(
            out,
            scale_factor = 2,
            mode="bilinear",
            align_corners = True)

        return self.cout(out + skip_out)