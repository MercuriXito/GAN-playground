#-*-coding:utf-8-*-

"""
    @file:			evaluate.py
    @autor:			Victor Chen
    @description:
        evaluate the model with several methods:
        * IS
        * FID
"""

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.models import inception_v3


def InceptionScore(images, device = None):
    """
    Calcualte InceptionSocre, 该函数需要和 Download/inception-score-pytorch-master/ 比对
    下正确性。
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = inception_v3(pretrained=True, aux_logits=False)
    model.to(device)

    dataloader = DataLoader(images, num_workers=4, batch_size= 32)

    outputs = []
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            images = nn.Upsample(size=299, mode="bilinear")(images)
            outputs.append(model(images))

        outputs = torch.cat(outputs, dim = 0)
        py = torch.sum(outputs, dim = 1)

        score = torch.mean(py * torch.log( outputs / py), dim=0)

    return score.sum().cpu()


class Inceptionv3Fe(nn.Module):
    def __init__(self, pretrained = True):
        super(Inceptionv3Fe, self).__init__()
        self.model = inception_v3(pretrained = pretrained,  aux_logits=False)
        modules = list(self.model.children())[:-1]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


def FrechetID(fakeimages, images, device = None):
    """
    Caculate the Frechet Inception Distance
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Inceptionv3Fe()
    model.to(device)

    out = []

    fakeloader = DataLoader(fakeimages, batch_size=32, num_workers=4)
    imageloader = DataLoader(images, batch_size=32, num_workers=4)
    upsampler = nn.Upsample(size=299, mode="nearest")

    with torch.no_grad():
        fout = []
        for fake in fakeloader:
            fake = upsampler(fake).to(device)
            fout.append(model(fake))

        out = []
        for images in imageloader:
            images = upsampler(images).to(device)
            out.append(model(images))

        out = torch.cat(out, dim = 0)
        tout = torch.cat(fout, dim = 0)

        fid = torch.norm(out.mean(dim = 0) - tout.mean(dim = 0))

        def covariance(X):
            X = X - mean(dim = 0)
            return torch.matmul(X.transpose(), X)
            
        cov1 = covariance(out)
        cov2 = covariance(tout)

        fid += torch.trace(cov1 + cov2 - torch.sqrt(torch.matmul(cov1, cov2)))

    return fid.cpu()