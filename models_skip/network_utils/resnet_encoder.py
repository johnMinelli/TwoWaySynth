# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, nz=200, dropout=False, pretrained=True):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 128, 256, 512])
        self.n_layers = np.array([64*64*64, 128*32*32, 256*16*16, 512*8*8])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.fc_modules = []
        for size in self.n_layers:
            fc = [nn.Linear(size, nz)]  # FIXME per adesso lo spiaccico a 200 fisso
            if dropout: fc += [nn.Dropout(0.3)]
            fc = nn.Sequential(*fc).to(torch.device('cuda'))
            self.fc_modules.append(fc)

    def forward(self, input_image):
        self.features = []
        b = input_image.size(0)
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.features.append(self.encoder.layer1(self.encoder.maxpool(x)))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))
        return [self.fc_modules[i](f.view(b, -1)) for i, f in enumerate(self.features)]
