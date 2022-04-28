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
from models_skip.network_utils.networks import get_non_linearity


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, nz, scales=range(4), num_output_channels=1, use_skips=True, dropout=False):
        super(DepthDecoder, self).__init__()
        nl_layer = get_non_linearity(layer_type='lrelu')

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.fc_modules = []
        for size in self.num_ch_enc:
            fc = [nn.Linear(nz, size)]  #FIXME per adesso lo spiaccico a 200 fisso
            if dropout: fc += [nn.Dropout(0.3)]
            fc += [nl_layer()]
            fc = nn.Sequential(*fc).to(torch.device('cuda'))
            self.fc_modules.append(fc)

        # decoder
        self.convs = OrderedDict()
        for i in range(3, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}
        b = input_features[0].size(0)

        input_features_recovered = []
        for i in range(len(self.num_ch_enc)):
            input_features_recovered.append(self.fc_modules[i](input_features[i]).view(b, self.num_ch_enc[i], 1, 1))
        # decoder
        x = input_features_recovered[-1]
        for i in range(3, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features_recovered[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs
