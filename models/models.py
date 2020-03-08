#-*-coding:utf-8-*-

"""
    @file:			models.py
    @autor:			Victor Chen
    @description:
        GAN 模型
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
import math

from functools import reduce


# Define Basic Blocks

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


def FCBlocks(in_features, out_features, bias = True, 
    norm_layer = None, activation = None, without_norm_layer = False):

    if norm_layer is None:
        norm_layer = nn.BatchNorm1d
    
    blocks = [nn.Linear(in_features, out_features, bias)]
    if not without_norm_layer:
        blocks.append(norm_layer(out_features))
    if activation is None:
        blocks.append(nn.ReLU(True))

    return blocks


def Conv2dBlock(in_channels, out_channels, kernel_size, stride, padding, bias = False,
    norm_layer = None, activation = None, without_norm_layer = False):

    if norm_layer is None:
        norm_layer = nn.BatchNorm2d

    blocks = [ nn.Conv2d(in_channels, out_channels, kernel_size, stride,
            padding = padding, bias = bias) ]
    if not without_norm_layer:
        blocks.append(norm_layer(out_channels))
    if activation is None:
        blocks.append(nn.LeakyReLU(0.2, True))
    return blocks


def Conv2dTransposeBlocks(in_channels, out_channels, kernel_size, stride, padding, bias = False,
    norm_layer = None, activation = None, without_norm_layer = False):

    if norm_layer is None:
        norm_layer = nn.BatchNorm2d

    blocks = [ nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
            stride, padding=padding, bias=bias) ]
    if not without_norm_layer:
        blocks.append(norm_layer(out_channels))
    if activation is None:
        blocks.append(nn.ReLU(True))
    return blocks


#####################
# Vanilla GAN with MLP for both Generator and Discriminator 

class MLPGenerator(nn.Module):
    def __init__(self, out_channels = 3, input_size = 100, output_size = 64):
        super(MLPGenerator, self).__init__()

        self.out_channels = out_channels
        config = [input_size, 8 * 8 * 8, 4 * 16 * 16, 32 * 32 * out_channels ]
        if output_size == 64:
            config.append(64 * 64 * out_channels)

        blocks = []
        last_layer = config[0]
        for i in range(1, len(config) - 1):
            this_layer = config[i]
            blocks += FCBlocks(last_layer, this_layer)
            last_layer = this_layer
        
        blocks += [
            nn.Linear(last_layer, config[-1]),
            nn.Tanh()
        ]
        self.generator = nn.Sequential(*blocks)
        self.output_size = output_size

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.generator(x)
        return x.view(x.size(0), self.out_channels , self.output_size, self.output_size)
    
class MLPDiscriminator(nn.Module):
    def __init__(self, in_channels, input_size):
        super(MLPDiscriminator, self).__init__()

        num_layers = 4
        blocks = []
        for i in range(num_layers - 1):
            cur_size = input_size // (2 ** i)
            cur_size = cur_size ** 2

            blocks += [
                nn.Linear(cur_size , cur_size // 4 ),
                nn.LeakyReLU(0.2)
            ]

        cur_size = cur_size // 4
        blocks += [
            nn.Linear(cur_size, 1 ),
            nn.Sigmoid()
        ]
        self.features = nn.Sequential(*blocks)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)


#########################
## DCGAN
## 

class DCGANG(nn.Module):
    def __init__(self, input_size = 100, out_channels=3, output_size = 64):
        """ 将 noise reshape 成 noise_size x 1x1 大小，初始通道数为512，
        每次卷积图像大小翻倍，通道数减半，直到输出指定大小
        """
        super(DCGANG, self).__init__()
        num_layers = int(math.log2(output_size * 1.0))
        last_channels = input_size
        blocks = []
        for i in range(num_layers - 1):
            this_channels = 2 ** (9 - i)
            blocks += Conv2dTransposeBlocks(last_channels, this_channels, 4, 2, 1)
            last_channels = this_channels
        blocks += [
            nn.ConvTranspose2d(last_channels, out_channels, 4, 2, 1),
            nn.Tanh()
        ]
        self.generator = nn.Sequential(*blocks)
        self.input_size = input_size

    def forward(self, x):
        x = x.view(x.size(0), self.input_size, 1, 1)
        return self.generator(x)


class DCGANC(nn.Module):
    def __init__(self, in_channels, input_size,  use_as_critic = False):
        super(DCGANC, self).__init__()

        without_norm_layer = False
        if use_as_critic:
            without_norm_layer = True

        num_layers = min(int(math.log2(input_size * 1.0)), 5)
        last_channels = in_channels
        blocks = []
        for i in range(num_layers):
            this_channels = 32 * ( 2 ** i)
            blocks += Conv2dBlock(last_channels, this_channels, 3, 2, 1, 
                without_norm_layer = without_norm_layer)
            last_channels = this_channels

        self.features = nn.Sequential(*blocks)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = [nn.Linear(last_channels, 1)]
        if not use_as_critic:
            self.classifier.append(nn.Sigmoid())
        self.classifier = nn.Sequential(*self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x).view(x.size(0), -1)
        return self.classifier(x)


#####################
## SA-GAN: consist of SpectralNorm, AttetionLayer.

class AttetionLayer(nn.Module):
    def __init__(self, in_channels):

        self.fconv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.gconv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.hconv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.vconv = nn.Conv2d(in_channels //8, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

        self.gamma = nn.Parameter(torch.tensor([0], dtype=torch.float32))
    
    def forward(self, x):

        B,C,W,H = x.size()

        fact = self.fconv(x).view(B, -1, W*H).permute(0, 2, 1)
        gact = self.gconv(x).view(B, -1, W*H)

        attention = self.softmax(torch.bmm(fact, gact))

        hact = self.hconv(x).view(B, -1, W*H)
        x0 = torch.bmm(hact, attention)

        x0 = self.vconv(x0)

        out = x0 * self.gamma + x
        
        return out


class SnLinear(nn.Module):
    """Linear Layer with spectral normalization.
        $${
            WSN = \frac{W}{\sigma{W}}
        }$$

    In each forward, the layer would apply spectral normalization on weight, so
    restict the weight with Lip-1 condition.
    """
    def __init__(self, in_features, out_features, bias = True):
        super(SnLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer = nn.Linear(in_features, out_features, bias)
        self.power_iteration = 1
        self.update_mode = False
        self.weight = self.layer.weight
        self.bias = self.layer.bias

    def _norm(self):

        try:
            u = getattr(self, "u")
            v = getattr(self, "v")

        except AttributeError:
            u = torch.rand([self.out_features, 1], dtype=torch.float32, requires_grad=False)
            v = torch.rand([self.in_features, 1], dtype=torch.float32, requires_grad=False)
            u = nn.Parameter(u)
            v = nn.Parameter(v)
            setattr(self, "u", u)
            setattr(self, "v", v)

        W = self.weight
        for i in range(self.power_iteration):
            vout = torch.matmul(W.transpose(0,1), u)
            vout = torch.div(vout, torch.norm(vout))
            uout = torch.matmul(W, v)
            uout = torch.div(uout, torch.norm(uout))

            specW = torch.matmul(uout.transpose(0,1), torch.matmul(W, vout) )

            self.layer.weight.data = self.layer.weight.data / specW


    def forward(self, x):
        self._norm()
        return self.layer(x)


class SnConv2d(nn.Module):
    """ Conv2d or Conv2dTranspose layer with Spectral Normalization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding=0, 
            bias = True, transpose = False, power_iteration = 1):

        super(SnConv2d, self).__init__()
        self.transpose = transpose
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if isinstance(self.kernel_size, (tuple, list)):
            mul = reduce(lambda x,y: x*y, self.kernel_size)
        elif isinstance(self.kernel_size, (int)):
            mul = self.kernel_size ** 2

        if self.transpose:
            self.module = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                stride, padding, bias = bias)
            self.m, self.n = self.in_channels, self.out_channels * mul
        else:
            self.module = nn.Conv2d(in_channels, out_channels, kernel_size, 
                stride, padding, bias = bias)
            self.m, self.n = self.out_channels, self.in_channels * mul
 
        self.power_iteration = power_iteration

        u = nn.Parameter(torch.rand([self.m, 1], requires_grad=False))
        v = nn.Parameter(torch.rand([self.n, 1], requires_grad=False))
        setattr(self, "u", u)
        setattr(self, "v", v)

    def _norm(self):
        u = getattr(self, "u")
        v = getattr(self, "v")

        W = self.module.weight.view(self.m, self.n)
        for _ in range(self.power_iteration):
            vout = torch.matmul(W.transpose(0,1), u)
            vout = torch.div(vout, torch.norm(vout))
            uout = torch.matmul(W, v)
            uout = torch.div(uout, torch.norm(uout))

            specW = torch.matmul(uout.transpose(0,1), torch.matmul(W, vout) )
            self.module.weight.data /= specW

    def forward(self, x):
        self._norm()
        return self.module(x)


#############################
### try modify the Discriminator in DCGAN with SpectralNormLayer
### without tactic in WGAN 

class SnDiscriminator(nn.Module):
    def __init__(self, in_channels):
        """ Use Conv Layer with Spectral Normalization Version
        """
        super(SnDiscriminator, self).__init__()

        def Conv2dBlock(in_channels, out_channels, \
            kernel_size, stride, padding, bias = False):

            return [
                SnConv2d(in_channels, out_channels, kernel_size, stride,
                    padding = padding, bias = bias, transpose=False),
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

