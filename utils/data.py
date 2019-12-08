#-*-coding:utf-8-*-

"""
    @file:			data.py
    @autor:			Victor Chen
    @description:
        load serveral dataset， 加载数据集
"""

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
import torchvision.transforms as transforms

def load_minst():
    """ load trainset from MNIST, preprocess contains:
    Resize to (64x64), Normalize
    """

    root = "/home/victorchen/workspace/Venus/"
    trainset = MNIST(root, train=True, transform=transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ]))

    loader_params = {
        "num_workers": 2,
        "shuffle": True,
        "batch_size": 32
    } 

    train = DataLoader(trainset, **loader_params)

    return train


def load_CIFAR10():
    """ load trainset from CIFAR10, preprocess contains:
    Resize to (64x64), Normalize 
    """
    root = "/home/victorchen/workspace/Venus/"
    trainset = CIFAR10(root, train=True, transform=transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
    ]))


    loader_params = {
        "num_workers": 2,
        "shuffle": True,
        "batch_size": 32
    } 

    train = DataLoader(trainset, **loader_params)


    return train


def load_Anime_faces():
    """ load Anime faces dataset(http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/)
    """

    root = "/home/victorchen/workspace/Venus/animeface-character-dataset/"

    dataset = ImageFolder(root + "thumb/", transform=transforms.Compose([
        transforms.CenterCrop(120),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
    ]))

    loader_params = {
        "num_workers": 2,
        "shuffle": True,
        "batch_size": 32
    } 

    train = DataLoader(dataset, **loader_params)

    return train
