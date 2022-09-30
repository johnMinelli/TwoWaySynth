# Encoder class Copyright Niantic 2019. Patent Pending. All rights reserved. https://github.com/nianticlabs/monodepth2

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, nz=200, dropout=False, pretrained=True):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        # add bottleneck ch size
        self.num_ch_enc = np.append(self.num_ch_enc, int(self.num_ch_enc[-1]/2))  # half used as feature and half compressed with fc

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](weights=ResNet18_Weights.DEFAULT if pretrained else None)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4
        final_dim = np.exp2(np.log2(self.num_ch_enc.max())-np.log2(self.num_ch_enc.min())).astype(np.int)
        fc = [nn.Linear(self.num_ch_enc[-1]*final_dim*final_dim, nz)]  # 256*8*8

        if dropout: fc += [nn.Dropout(0.3)]
        self.fc = nn.Sequential(*fc)

    def to(self, device):
        self.fc.to(device)
        self.encoder.to(device)
        super().to(device)
        return self

    def forward(self, input_image):
        self.features = []
        # x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))
        # apply bottleneck in channel dimension splitting last feature
        self.features += torch.split(self.features[-1], int(self.features[-1].size(1)-self.num_ch_enc[-1]), dim=1)  # ch_last feature - ch_bottleneck = 512 - 256
        latent_compressed_flattened = self.fc(self.features[-1].view(input_image.size(0),-1))

        return latent_compressed_flattened, self.features[:-1]
