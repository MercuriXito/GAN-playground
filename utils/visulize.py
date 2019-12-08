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
    images = make_grid(batches).detach().numpy().transpose((1,2,0))
    
    figure = plt.figure(figsize=figsize)
    plt.imshow(images)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return plt

def save_batch_images(batches, image_name, root = "./saved/images/"):
    save_image(batches, "{}{}.png".format(root, image_name))