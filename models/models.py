#-*-coding:utf-8-*-

"""
    @file:			models.py
    @autor:			Victor Chen
    @description:
        GAN 模型
"""

import torch
import torch.nn as nn

#####################
# Vanilla GAN with MLP for both Generator and Discriminator 

class MLPGenerator(nn.Module):
    def __init__(self, out_channels = 3, input_size = 100,):
        super(MLPGenerator, self).__init__()

        def LinearBlocks(in_channels, out_channels):
            return [
                nn.Linear(in_channels, out_channels),
                # nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]

        self.out_channels = out_channels

        self.generator = nn.Sequential(
            *LinearBlocks(input_size, 32 * 4 * 4),
            *LinearBlocks(32 * 4 * 4, 64 * 8 * 8),
            nn.Linear(64 *8 *8 , 64 * 64 * out_channels ),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.generator(x)
        return x.view(x.size(0), self.out_channels , 64, 64)
    
class MLPDiscriminator(nn.Module):
    def __init__(self, in_channels = 3):
        super(MLPDiscriminator, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(64 * 64 * in_channels, 64 * 64 ),
            nn.LeakyReLU(0.2),
            nn.Linear( 64 * 64, 32 * 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32 * 32, 8 * 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8*8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)


#########################
## DCGAN
## 

class DCGAN(nn.Module):
    def __init__(self, input_size = 100, out_channels=3):
        super(DCGAN, self).__init__()

        def Conv2dTransposeBlocks(in_channels, out_channels, 
            kernel_size, stride, padding, bias = False):
            return [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                    stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
        
        self.generator = nn.Sequential(
            *Conv2dTransposeBlocks(100, 512, 4, 1, 0),
            *Conv2dTransposeBlocks(512, 256, 5, 1, 0),
            *Conv2dTransposeBlocks(256, 128, 4, 2, 1),
            *Conv2dTransposeBlocks(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), 100, 1, 1)
        return self.generator(x)


class SimpleDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(SimpleDiscriminator, self).__init__()

        def Conv2dBlock(in_channels, out_channels, \
            kernel_size, stride, padding, bias = False):

            return [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                    padding = padding, bias = bias),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            ]

        self.features = nn.Sequential(
            *Conv2dBlock(in_channels, 32, 3, 2, 1),
            *Conv2dBlock(32, 64, 3, 2, 1),
            *Conv2dBlock(64, 128, 3, 2, 1),
            *Conv2dBlock(128, 256, 3, 2, 1),
            *Conv2dBlock(256, 512, 3, 2, 1),
        )

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x).view(x.size(0), -1)
        return self.classifier(x)

#########################
## slighted modified Discriminator for WGAN
## - remove sigmoide
## - with or without BatchNorm layer

class WGANcritic(nn.Module):
    def __init__(self, in_channels, with_bn = True):
        super(WGANcritic, self).__init__()

        def Conv2dBlock(in_channels, out_channels, \
            kernel_size, stride, padding, bias = False):
            if with_bn:
                return [
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                        padding = padding, bias = bias),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2)
                ]
            else:
                return [
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                        padding = padding, bias = bias),
                    nn.LeakyReLU(0.2)
                ]

        self.features = nn.Sequential(
            *Conv2dBlock(in_channels, 32, 3, 2, 1),
            *Conv2dBlock(32, 64, 3, 2, 1),
            *Conv2dBlock(64, 128, 3, 2, 1),
            *Conv2dBlock(128, 256, 3, 2, 1),
            *Conv2dBlock(256, 512, 3, 2, 1),
        )

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(512, 1) # without sigmoid 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x).view(x.size(0), -1)
        return self.classifier(x)