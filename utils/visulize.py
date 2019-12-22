#-*-coding:utf-8-*-

"""
    @file:			visulize.py
    @autor:			Victor Chen
    @description:
        utils of visulization, 可视化工具
"""

import os,sys

import numpy as np 
import matplotlib.pyplot as plt

import torch
from torchvision.utils import make_grid, save_image

def show_batch_images(batches, figsize = (15,15), title="images"):
    images = make_grid(batches, nrow=16).detach().cpu().numpy().transpose((1,2,0))
    
    figure = plt.figure(figsize=figsize)
    plt.imshow(images)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return plt


def show_retrived_batch_images(batches, means, stds, figsize = (15,15), title="images"):

    batch_size, num_channel = batches.size()[:2]
    nrow = int(batch_size ** 0.5) 

    images = make_grid(batches, nrow).detach().cpu().numpy().transpose((1,2,0))

    for channel in range(num_channel):
        images[:,:,channel] = ( images[:,:,channel] * stds[channel] + means[channel] ) * 255.0
        
    images = images.astype(int)
    
    figure = plt.figure(figsize=figsize)
    plt.imshow(images)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return plt

def save_batch_images(batches, image_name, root = "./saved/images/"):
    save_image(batches, "{}{}.png".format(root, image_name), nrow=16)

def save_retrived_batch_images(batches, means, stds, image_name, root = "./saved/images/"):

    batch_size, num_channel = batches.size()[:2]
    nrow = int(batch_size ** 0.5) 

    with torch.no_grad():
        for channel in range(num_channel):
            batches[:,:,channel] = (batches[:,:,channel] * stds[channel]
                 + means[channel]) * 255.0
    
        batches = torch.as_tensor(batches, dtype=torch.int)

    save_image(batches, "{}{}.png".format(root, image_name), nrow)