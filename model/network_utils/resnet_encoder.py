# Encoder class Copyright Niantic 2019. Patent Pending. All rights reserved. https://github.com/nianticlabs/monodepth2

from __future__ import absolute_import, division, print_function

import numpy as np
import torch.nn as nn
import torchvision.models as models

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, nz=200, dropout=False, pretrained=True):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

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
        final_dim = np.exp2(np.log2(self.num_ch_enc.max())-np.log2(self.num_ch_enc.min())).astype(np.int)
        fc = [nn.Linear(self.num_ch_enc[-1]*final_dim*final_dim, nz)]  # 512*8*8
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

        return self.fc(self.features[-1].view(input_image.size(0),-1)), self.features[:-1]
        # MOD here FC instead of MaxPool: I don't want to classify, the details must be maintained.
