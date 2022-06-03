# Decoder class Copyright Niantic 2019. Patent Pending. All rights reserved. https://github.com/nianticlabs/monodepth2

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from collections import OrderedDict

from model.network_utils.layers import *
from model.network_utils.networks import get_non_linearity, get_norm_layer


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, nz=200, scales=range(5), num_output_channels=1, dropout=False, norm_layer_type='batch', nl_layer_type='lrelu', upsample_mode='bilinear'):
        super(DepthDecoder, self).__init__()
        norm_layer = get_norm_layer(norm_type=norm_layer_type)
        nl_layer = get_non_linearity(layer_type=nl_layer_type)
        self.upsample_mode = upsample_mode

        fc = [nn.Linear(nz, num_ch_enc[-1]*8*8)]
        if dropout: fc += [nn.Dropout(0.3)]
        fc += [nl_layer()]
        self.fc = nn.Sequential(*fc)

        self.num_output_channels = num_output_channels
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out, norm_layer, nl_layer)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            num_ch_out = self.num_ch_dec[i]
            if i > 0:
                self.convs[("upconv_s", i, 1)] = ConvBlock(num_ch_in + self.num_ch_enc[i - 1], num_ch_out, norm_layer, nl_layer)
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out, norm_layer, nl_layer)
        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = []

        if isinstance(input_features, list):
            x = input_features[-1]
            use_skips = True
        else:
            x = self.fc(input_features).view(input_features.size(0),self.num_ch_enc[-1],8,8)  # MOD here insteaa of conclutional blocks to upsample there is a FC
            use_skips = False

        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = upsample(x, mode=self.upsample_mode)
            if use_skips and i > 0:
                x = [x, input_features[i - 1]]
                x = torch.cat(x, 1)
                x = self.convs[("upconv_s", i, 1)](x)
            x = self.convs[("upconv", i, 1)](x)         # MOD this is fixed instead in an else branch
            if i in self.scales:
                self.outputs += [self.convs[("dispconv", i)](x)]
        return self.outputs                             # MOD here consider that even if fetures returned have dim in range [16-256] they were downscaled to [8-128]
