import torch

from options import train_config_parser
from utils import data
from utils.Updater import WGANUpdater, Updater
from utils.frame import train
from utils.record import Logger
from models.models import DCGANC, DCGANG, MLPDiscriminator, MLPGenerator


def main():

    opt = train_config_parser()
    model = opt.model
    dataset = opt.dataset
    device = opt.device

    print("Using parameters:")
    for name, value in dict(opt._get_kwargs()).items():
        print("%20s:\t%s" %(name, value))

    # choose dataset
    if dataset == "mnist":
        dataloader = data.load_minst(opt)
    elif dataset == "cifar10":
        dataloader = data.load_CIFAR10(opt)
    elif dataset == "animefaces":
        dataloader = data.load_Anime_faces(opt)
    else:
        raise Exception("This")
    
    # choose model
    channels = 1 if dataset == "mnist" else 3
    if model == "gan":
        netG = MLPGenerator(channels, opt.noise_size, opt.image_size)
        netD = MLPDiscriminator(channels, opt.image_size)
    elif "wgan" in model:
        netG = DCGANG(opt.noise_size, channels, opt.image_size)
        netD = DCGANC(channels, opt.image_size, use_as_critic=True)
    elif model == "dcgan":
        netG = DCGANG(opt.noise_size, channels, opt.image_size)
        netD = DCGANC(channels, opt.image_size)
    
    netG.to(opt.device)
    netD.to(opt.device)

    print("Generator:")
    print(netG)
    print("Discriminator:")
    print(netD)

    # x = netG(torch.randn(opt.batch_size, opt.noise_size, device = opt.device))
    # print(x.size())
    # print(opt.image_size)
    # print(netD(x).size())
    # exit(0)

    # choose update method
    if "wgan" in model:
        update_class = WGANUpdater
    else:
        update_class = Updater

    logger = Logger(opt)
    train(opt, netG, netD, dataloader, update_class, logger)

if __name__ == "__main__":
    main()