import numpy as np
import torch 
import torch.nn as nn
import torch.autograd as autograd

import time

def weight_clip(net, clip_range):
    for param in net.parameters():
        param.data.clamp_(-clip_range, clip_range)
        
def close_grad(net):
    for param in net.parameters():
        param.requires_grad_(False)

def open_grad(net):
    for param in net.parameters():
        param.requires_grad_(True)
    
def calculate_gradient_penalty(netD, images, fake_images, gp_lambda, device):

    batch_size, C, W, H = images.size()
    alpha = torch.randn((batch_size, 1), device=device)
    alpha = alpha.expand((batch_size, C * W * H)).contiguous()
    alpha = alpha.view_as(images)

    interpolate = alpha * images + (1 - alpha) * fake_images
    interpolate = interpolate.to(device)
    interpolate.requires_grad_(True)

    out = netD(interpolate)

    grads = autograd.grad(out, interpolate, 
        grad_outputs=torch.ones_like(out).type(torch.float).to(device),
        retain_graph=True, create_graph=True)[0]

    grads = grads.view(grads.size(0), -1)
    return gp_lambda * ((grads.norm(p=2, dim = 1) - 1) ** 2).mean()


def get_timestr():
    format = "%Y_%m_%d_%H_%M_%S"
    return time.strftime(format, time.localtime())