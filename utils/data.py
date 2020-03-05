#-*-coding:utf-8-*-

"""
    @file:			data.py
    @autor:			Victor Chen
    @description:
        load serveral dataset， 加载数据集
"""

import os,sys
import PIL.Image as Image

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
import torchvision.transforms as transforms


def load_minst(opt):
    """ load trainset from MNIST, preprocess contains:
    Resize to (64x64), Normalize
    """

    root = opt.data_root
    image_size = opt.image_size
    mean, std = [0.5], [0.5]
    trainset = MNIST(root, train=True, transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ]))

    setattr(opt, "mean", mean)
    setattr(opt, "std", std)

    loader_params = {
        "num_workers": opt.num_workers,
        "shuffle": opt.shuffle,
        "batch_size": opt.batch_size
    } 

    train = DataLoader(trainset, **loader_params)

    return train


def load_CIFAR10(opt):
    """ load trainset from CIFAR10, preprocess contains:
    Resize to (64x64), Normalize 
    """
    root = opt.data_root
    image_size = opt.image_size
    mean, std = [0.5, 0.5, 0.5],[0.5, 0.5, 0.5]

    trainset = CIFAR10(root, train=True, transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]))

    setattr(opt, "mean", mean)
    setattr(opt, "std", std)

    loader_params = {
        "num_workers": opt.num_workers,
        "shuffle": opt.shuffle,
        "batch_size": opt.batch_size
    } 

    train = DataLoader(trainset, **loader_params)

    return train


def load_Anime_faces(opt):
    """ Anime faces dataset from (http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/)
    """

    root = opt.data_root
    image_size = opt.image_size
    mean, std = [0.5, 0.5, 0.5],[0.5, 0.5, 0.5]

    dataset = ImageFolder(root + "thumb/", transform=transforms.Compose([
        transforms.CenterCrop(120),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]))

    setattr(opt, "mean", mean)
    setattr(opt, "std", std)

    loader_params = {
        "num_workers": opt.num_workers,
        "shuffle": opt.shuffle,
        "batch_size": opt.batch_size
    } 

    train = DataLoader(trainset, **loader_params)

    return train


class CelebA(Dataset):
    """ CelebA Dataset, using the aligned and croped version.
        http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    """
    def __init__(self, root = "/home/victorchen/workspace/Venus/celebA/", transform = None , target_transform = None):

        self.img_dir = root + "images/"
        identity_file = root + "Anno/identity_CelebA.txt"

        with open(identity_file, "r") as f:
            self.identitiy = f.readlines()

        self.transform = transform
        self.target_transform = target_transform
        self.channel_mean = [0.50003925, 0.42008925, 0.37377534]
        self.channel_std = [0.30878809, 0.28794379, 0.28661432]

    def __len__(self):
        return len(self.identitiy)

    def __getitem__(self, idx):

        image, label = self.identitiy[idx].replace("\r","").replace("\n","").split()

        image = Image.open(self.img_dir + image)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
        
        
def load_celebA():
    """ default load celebA dataset
    """
    channel_mean = [0.50003925, 0.42008925, 0.37377534]
    channel_std = [0.30878809, 0.28794379, 0.28661432]

    dataset = CelebA(transform=transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(channel_mean, channel_std)
    ]))

    setattr(dataset, "mean", channel_mean)
    setattr(dataset, "std", channel_std)

    loader_params = {
        "num_workers": 2,
        "shuffle": True,
        "batch_size": 128
    } 

    train = DataLoader(dataset, **loader_params)

    return train
