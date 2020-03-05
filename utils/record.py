#-*-coding:utf-8-*-

"""
    @file:			record.py
    @autor:			Victor Chen
    @description:
        use tensorboardX to record training details
"""

import os,sys
import random
import numpy as np 
import matplotlib.pyplot as plt
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

import utils.functions as F
from collections import Iterable


def _plt_imshow(narray, name, is_gray = False):

    if is_gray:
        plt.imshow(narray, cmap = plt.cm.gray_r)
    else:
        plt.imshow(narray)

    plt.xticks([])
    plt.yticks([])
    plt.title(name)
    return plt


def _get_grid(images, mean = None, std = None, nrow = 0, 
    retrieve = True):

    if nrow < 1:
        nrow = 8

    grid = make_grid(images, nrow=nrow).detach().cpu().numpy().transpose(1,2,0)
    if retrieve:
        if mean is not None and std is not None:
            # retreive the images
            for channel in range(images.size(1)):
                grid[:,:,channel ] = (grid[:,:,channel] * std[channel] + mean[channel]) * 255.0
        else:
            grid = grid * 255.0
        grid = grid.astype(np.uint8)

    return grid
    

def show_batch(images, mean = None, std = None, nrow = 0,
        title = "images", auto_select_nrow = False, retrieve = True ):

    if auto_select_nrow:
        nrow = int(images.size(0) ** 0.5)

    grid = _get_grid(images, mean, std, nrow, retrieve)
    _plt_imshow(grid, "Images", images.size(1) == 1).show()


def save_batch(images, root, name, mean = None, std = None, nrow = 0, 
        auto_select_nrow = False, retrieve = True):

    if auto_select_nrow:
        nrow = int(images.size(0) ** 0.5)

    grid = _get_grid(images, mean, std, nrow, retrieve)
    pltt = _plt_imshow(grid, "Images", images.size(1) == 1)
    pltt.savefig(root + name)
    

def show_transition(netG, num_img = 8, num_transition = 8, noise_size = 100,
        means = None, stds = None, device = None, savename = None):

    if device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fromx = torch.randn((num_img, noise_size), device=device)
    tox = torch.randn((num_img, noise_size), device=device)
    netG.to(device)

    # interpolate
    minus = (tox - fromx) / num_transition
    xlist = [fromx]
    for i in range(num_transition - 1):
        xlist.append(fromx + (i+1)*minus)

    x = torch.cat(xlist, dim = 0).contiguous()
    images = netG(x)

    show_batch(images, means, stds, nrow=num_img)

    if savename is not None:
        # save_batch(images, savename, mean=means, std=stds, nrow=num_img)
        pass


class Logger:
    def __init__(self, opt):

        self.base = opt.save_root
        self.root = self.base  # root for saving data
        self.model_root = self.root + "models/"
        self.images_root = self.root + "images/"

        if not os.path.exists(self.base):
            os.makedirs(self.base)

        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.writer = SummaryWriter(log_dir=self.root)
        self.opt = opt
        self.scalars = {}
        self.images = {}
        self.nrow = opt.save_images_nrow


    def save_model(self, netG, netD, epoch):

        if not os.path.exists(self.model_root):
            os.makedirs(self.model_root)

        # save models
        for tag, model in zip(["G","D"],[netG, netD]):
            name = "{}_{}_{}.pth".format(self.opt.model, tag, self.opt.dataset)
            f = self.model_root + name
            torch.save(model.state_dict(), f)

        # save the parameter
        opt_dict = dict(self.opt._get_kwargs())
        opt_dict["trained_epoch"] = epoch
        dict_name = "{}_{}.json".format(self.opt.model, self.opt.dataset)
        with open(self.model_root + dict_name, "w") as f:
            json.dump(opt_dict, f)


    # def save_images(self, images, name):

        # if not os.path.exists(self.images_root):
            # os.makedirs(self.images_root)

        # image_name = "{}.png".format(name)
        # batch_size = self.opt.batch_size
        # mean, std = self.opt.mean, self.opt.std
        # nrow = self.nrow
        # save_batch(images, self.images_root, image_name, mean, std, nrow, retrieve=(images.size(1) != 1))


    def _save_writer_images(self, tag, images, step):

        mean, std = self.opt.mean, self.opt.std
        nrow = self.nrow
        grid = _get_grid(images, mean, std, nrow, retrieve= (images.size(1) != 1) )
        self.writer.add_image(tag, grid, global_step=step, dataformats="HWC")
        self.writer.flush()

    def add_images(self, tag, tensor):
        """ Temporarily save images, further use `write_images` to actually 
        write images physically.
        """
        self.images[tag] = tensor

    def write_images(self, step, tags = None): # use tags to specific the images to write
        if tags is None:
            for tag, value in self.images.items():
                self._save_writer_images(tag, value, step)
        else:
            if not isinstance(tags, list):
                tags = [tags]
            for tag in tags:
                self._save_writer_images(tag, self.images[tag], step)

    def add_scalars(self, tag, val):
        self.scalars[tag] = val

    def write_scalars(self, step, tags = None): # use tags to specific the scalar to write
        if tags is None:
            for tag, value in self.scalars.items():
                self.writer.add_scalar(tag, value, global_step= step)
        else:
            if not isinstance(tags, list):
                tags = [tags]
            for tag in tags:
                self.writer.add_scalar(tag, self.scalars[tag], step)


if __name__ == "__main__":
    pass