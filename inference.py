import torch 

from utils import data
from models.models import DCGANC, DCGANG, MLPDiscriminator, MLPGenerator
from options import inference_config_parser
from utils.record import show_batch
from utils.frame import Noise


def main():

    opt = inference_config_parser()
    model = opt.model
    dataset = opt.dataset
    device = opt.device

    print("Using parameters:")
    for name, value in dict(opt._get_kwargs()).items():
        print("%20s:\t%s" %(name, value))
    
    # choose model
    channels = 1 if dataset == "mnist" else 3
    if model == "gan":
        netG = MLPGenerator(channels, opt.noise_size, opt.image_size)
    elif "wgan" in model:
        netG = DCGANG(opt.noise_size, channels, opt.image_size)
    elif model == "dcgan":
        netG = DCGANG(opt.noise_size, channels, opt.image_size)
    
    netG.to(opt.device)

    print("Generator:")
    print(netG)

    if opt.show_transition:
        show_transition(opt, netG)
        return

    batch_size = opt.gen_batch
    nrow = opt.gen_images_nrow
    noise_generator = Noise(opt.noise_size, opt.device)
    z = noise_generator(batch_size)

    images = netG(z)
    show_batch(images, opt.mean, opt.std, nrow, retrieve=(images.size(1) != 1))


def show_transition(opt, netG):

    batch_size = opt.gen_batch
    nrow = batch_size
    noise_generator = Noise(opt.noise_size, opt.device)

    z1 = noise_generator(batch_size)
    z2 = noise_generator(batch_size)

    zs = []
    number = opt.transition_number

    zstrip = (z2 - z1) / (number - 1)
    for i in range(number - 1):
        zt = z1 + i * zstrip
        zs.append(zt)
    zs.append(z2)
    zs = torch.cat(zs, dim = 0).to(opt.device)

    images = netG(zs)
    show_batch(images, opt.mean, opt.std, nrow, retrieve=(images.size(1) != 1))
    

if __name__ == "__main__":
    main()