from .Layers import BidirectionalLSTM
from .Encoder import Encoder
from .Decoder import Decoder
from torch import nn

class CUnet(nn.Module):
    def __init__(self, cfg):
        super(CUnet, self).__init__()
        
        self.encoder = Encoder()
        self.rnn = nn.Sequential(
            *[BidirectionalLSTM(512, 512, 1024) for _ in range(cfg.rnn_layers)]
        )
        self.decoder = Decoder(cfg)
        
        self.out = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x, connections = self.encoder(x)
        x = self.rnn(x)
        x = self.decoder(x, connections)
        return self.out(x)