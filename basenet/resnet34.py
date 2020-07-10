import os
from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models


class Resnet34(torch.nn.Module):
    def __init__(self, pretrained=True, freeze=False):
        super().__init__()
        backbone = nn.Sequential(*list(models.resnet34(pretrained).children())[:-2-1]) # drop last block as in VGG

        self.slice1 = torch.nn.Sequential(*backbone[:4])    # [B, 64,   H/2,   W/2]
        self.slice2 = torch.nn.Sequential(*backbone[4:5])   # [B, 64,   H/4,   W/4]
        self.slice3 = torch.nn.Sequential(*backbone[5:6])   # [B, 128,  H/8,   W/8]
        self.slice4 = torch.nn.Sequential(*backbone[6:])    # [B, 256,  H/16,  W/16]

        # fc6, fc7 without atrous conv
        self.slice5 = torch.nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(512, 512, kernel_size=1),
        )

        if freeze:
            for param in self.slice1.parameters():  # only first conv
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        outputs = namedtuple("Outputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out
