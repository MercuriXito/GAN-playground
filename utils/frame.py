import numpy as np
import torch 
import torch.optim as optim
import time
from tqdm import tqdm

class Noise:
    def __init__(self, noise_size, device):
        self.noise_size = noise_size
        self.device = device

    def __call__(self, batch_size):
        return torch.randn((batch_size, self.noise_size), device = self.device)


def get_optimizer(name, net, lr, betas, momentum):
    if name == "adam":
        optimizer = optim.Adam(net.parameters(), lr = lr, betas = betas)
    elif name == "sgd":
        optimizer = optim.SGD(net.parameters(), lr = lr, momentum= momentum)
    else:
        raise Exception("No")
    return optimizer


def train(opt, netG, netD, dataloader, update_class, logger):
    batch_size = opt.batch_size
    number = opt.save_num_image - 1
    noise_size = opt.noise_size
    device = opt.device
    epochs = opt.epochs
    train_D_interval = opt.D_train_interval
    train_G_interval = opt.G_train_interval
    interval = opt.save_model_interval

    i_s = len(dataloader.dataset) // batch_size
    step_is = list(np.linspace(0, i_s, number, dtype = int))

    lr_G = opt.G_lr
    lr_D = opt.D_lr
    optimizerD = get_optimizer(opt.D_optimizer, netD, lr_D, 
        opt.adam_betas, opt.sgd_momentum)
    optimizerG = get_optimizer(opt.G_optimizer, netG, lr_G, 
        opt.adam_betas, opt.sgd_momentum)

    noise_generator = Noise(noise_size, device)
    updater = update_class(opt, netG, netD, optimizerD, optimizerG, noise_generator, logger)
    
    # from utils.record import Logger
    # logger = Logger(opt)

    Dstep = 0
    Gstep = 0
    record_step = 0
    starttime = time.clock()
    for epoch in range(1, epochs + 1):
        print("Epoch:{}/{}".format(epoch, epochs))
        for i, (images, _) in enumerate(tqdm(dataloader)):
            # updat D
            if i % train_D_interval == 0:
                updater.updateD(images, Dstep) # updater save the loss it need
                Dstep += 1
            if i % train_G_interval == 0:
                updater.updateG(images, Gstep)
                Gstep += 1
            if i in step_is:
                # save the images
                logger.write_images(record_step)
                record_step += 1

        if epoch % interval == 0 or epoch == epochs:
            # save both models
            logger.save_model(netG, netD, epoch)

        logger.write_images(record_step)
        record_step += 1

    endtime = time.clock()
    train_time = (endtime - starttime)
    print("Training Using %5.2fs" %(train_time))
    logger.writer.close()
