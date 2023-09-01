import torch
import torch.nn as nn

from model.GANModel.base.MLPBaseLayer import MLPBaseLayer


class Generator(nn.Module):
    def __init__(self, hparams):
        super(Generator, self).__init__()

        self.hparams = hparams
        self.num_linear_layers = hparams.num_linear_layers
        self.in_channels = hparams.in_channels
        self.out_channels = hparams.out_channels
        self.ch_mult = hparams.ch_mult
        self.mid_channels = hparams.mid_channels

        self.layers = nn.ModuleList()
        channels = [self.in_channels, self.out_channels]
        for i in range(len(self.ch_mult)):
            channels.insert(1, self.ch_mult[i]*self.mid_channels)

        for i in range(len(channels) - 1):
            self.layers.append(MLPBaseLayer(n_linear=self.num_linear_layers,
                                            in_channels=channels[i],
                                            mid_channels=channels[i + 1],
                                            out_channels=channels[i + 1],
                                            activation=self.hparams.activation,
                                            normalization=self.hparams.normalization,
                                            dropout=self.hparams.dropout))
        # self.layers.append(nn.Softmax(dim=1))

    def forward(self, shape, device):
        h = torch.randn((shape[0], self.in_channels), device=device)
        for module in self.layers:
            h = module(h)
        return h

    def sample(self, shape, device):
        return self.forward(shape, device)
