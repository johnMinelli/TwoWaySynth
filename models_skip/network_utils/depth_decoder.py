# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from models_skip.network_utils.layers import *
from models_skip.network_utils.networks import get_non_linearity, get_norm_layer


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, nz=200, scales=range(5), num_output_channels=1, dropout=False):
        super(DepthDecoder, self).__init__()
        norm_layer = get_norm_layer(norm_type='batch')
        nl_layer = get_non_linearity(layer_type='lrelu')
        self.upsample_mode = 'bilinear'

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
            x = self.fc(input_features).view(input_features.size(0),self.num_ch_enc[-1],8,8)  # TODO qui fc invece che conv
            use_skips = False

        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = upsample(x, mode=self.upsample_mode)
            if use_skips and i > 0:
                x = [x, input_features[i - 1]]
                x = torch.cat(x, 1)
                x = self.convs[("upconv_s", i, 1)](x)
            x = self.convs[("upconv", i, 1)](x)         # TODO qui ho messo fisso questo invece che in else
            if i in self.scales:
                self.outputs += [self.convs[("dispconv", i)](x)]
        return self.outputs                             # TODO qui considera che anche se escono 16-256 vengono downscalati a 8-128
