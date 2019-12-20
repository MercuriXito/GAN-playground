#-*-coding:utf-8-*-

"""
    @file:			main.py
    @autor:			Victor Chen
    @description:
        train field
"""

import time
import numpy as np 
import matplotlib.pyplot as plt

import torch

from models.models import DCGAN, WGANcritic, MLPDiscriminator, MLPGenerator, SimpleDiscriminator
from utils.data import load_minst, load_CIFAR10, load_Anime_faces
from utils.WGAN import WGANTrainer
from utils.WGANGP import WGANGPTrainer
from utils.VanillaGAN import 
from utils.visulize import show_batch_images


class BatchNoiseGenerator:
    def __init__(self, noise_size = 100, mean = 0, std = 1):
        self.noise_size = noise_size
        self.mean = mean
        self.std = std

    def __call__(self, batch_size):
        x = ( torch.randn((batch_size, self.noise_size, 1, 1)) + self.mean ) * self.std
        return x


def train_example_vanillagan_mlp():
    """ example of training MLP Generator on MNIST with vanilla training method.
    """

    train = load_minst()

    noise_size = 100
    dis = MLPDiscriminator(in_channels = 1)
    gen = MLPGenerator(out_channels = 1, input_size = noise_size)
    noise_generator = BatchNoiseGenerator(noise_size= noise_size)

    trainer = GANTrainer(epochs = 40, lr_G = 0.0002,
             lr_D = 0.0002, folder_name = "MLP_Mnist")

    if torch.cuda.is_available():
        gen.to("cuda")
        dis.to("cuda")

    trainer.train(gen, dis, train, noise_generator)


def train_example_dcgan(dataset_name):
    """ An example of training DCGAN with vanilla training method.
    
    attr:
        dataset_name :  in range (MNIST, CIFAR10, AnimeFaces)
    """

    channels = 3
    if dataset_name == "AnimeFaces":
        train = load_Anime_faces()
    elif dataset_name == "CIFAR10":
        train = load_CIFAR10()
    elif dataset_name == "MNIST":
        train = load_minst()
        channels = 1
    else:
        raise Exception("No such supported dataset right now.")

    noise_size = 100
    gen = DCGAN(out_channels = channels, input_size = noise_size)
    dis = SimpleDiscriminator(in_channels = channels)
    noise_generator = BatchNoiseGenerator(noise_size = noise_size)

    trainer = GANTrainer(epochs = 200, lr_G = 0.0002, 
            lr_D = 0.0002, folder_name = "DCGAN_{}".format(dataset_name))

    if torch.cuda.is_available():
        gen.to("cuda")
        dis.to("cuda")

    trainer.train(gen, dis, train, noise_generator)


def train_example_wgan(dataset_name, gp_mode = True):
    """ An example of training DCGAN with WGAN training method.
    """

    channels = 3
    if dataset_name == "AnimeFaces":
        train = load_Anime_faces()
    elif dataset_name == "CIFAR10":
        train = load_CIFAR10()
    elif dataset_name == "MNIST":
        train = load_minst()
        channels = 1
    else:
        raise Exception("No such supported dataset right now.")

    noise_size = 100
    gen = DCGAN(out_channels = channels, input_size = noise_size)
    dis = WGANcritic(in_channels = channels, with_bn = not gp_mode)
    noise_generator = BatchNoiseGenerator(noise_size = noise_size)

    trainer = GANTrainer(epochs = 200, lr_G = 0.0001, 
            lr_D = 0.0001, folder_name = "WGAN{}_{}".format(
                "GP" if gp_mode else "", dataset_name))

    if torch.cuda.is_available():
        gen.to("cuda")
        dis.to("cuda")

    trainer.train(gen, dis, train, noise_generator)    


def main():
    train = load_Anime_faces()

    dis = WGANcritic(in_channels=3, with_bn = False)
    gen = DCGAN(out_channels=3)

    noise_generator = BatchNoiseGenerator()

    trainer = WGANGPTrainer(200, 0.0001, 0.0001, "AnimeFaces", "WGANGP-Anime")

    gen.to("cuda")
    dis.to("cuda")
    trainer.train(gen, dis, train, noise_generator)


if __name__ == "__main__":

    main()