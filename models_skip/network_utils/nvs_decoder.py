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


class NvsDecoder(nn.Module):
    def __init__(self, num_ch_enc, nz=200, scales=range(4), num_output_channels=3, dropout=False, norm_layer_type='instance', nl_layer_type='lrelu', upsample_mode='bilinear'):
        super(NvsDecoder, self).__init__()
        norm_layer = get_norm_layer(norm_type=norm_layer_type)
        nl_layer = get_non_linearity(layer_type=nl_layer_type)
        self.upsample_mode = upsample_mode

        self.num_output_channels = num_output_channels
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.final_dim = np.exp2(np.log2(self.num_ch_enc.max())-np.log2(self.num_ch_enc.min())).astype(np.int)

        fc = [nn.Linear(nz, num_ch_enc[-1]*self.final_dim*self.final_dim)]
        if dropout: fc += [nn.Dropout(0.3)]
        fc += [nl_layer()]
        self.fc = nn.Sequential(*fc)

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
                num_ch_in += self.num_ch_enc[i - 1]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out, norm_layer, nl_layer)  # following UNet you should do here a conv3x3,relu,conv3x3,relu and this is both output (no need of "predconv") and also input of next step)

        for s in self.scales:
            self.convs[("predconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels, use_refl=False)

        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_encoded, input_features):
        self.outputs = []

        x = self.fc(input_encoded).view(input_encoded.size(0), self.num_ch_enc[-1], self.final_dim, self.final_dim)  # TODO qui ho usato una fc larga invece di ConvTranspose2d(in, out, kernel_size=4, stride=2, padding=1) fino a 8x8

        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x, mode=self.upsample_mode)]
            if i > 0:
                x += [input_features[i-1]]
            x = torch.cat(x, 1)                 # TODO qui non skippo con le mid output scales sarebbe cat([layer(n,256,256), outpred(3,256,256), skipfeature(n,256,256)])
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs += [self.convs[("predconv", i)](x)]
        return self.outputs
