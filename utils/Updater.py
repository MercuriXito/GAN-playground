import numpy as np
import torch 
import torch.nn as nn
import torch.autograd as autograd

from utils.functions import weight_clip, close_grad, open_grad, calculate_gradient_penalty
from utils.record import Logger


class WGANUpdater:
    def __init__(self, opt, netG, netD, optimizerD, optimizerG, noise_generator,
        logger):
        self.G = netG
        self.D = netD
        self.optimizerD = optimizerD
        self.optimizerG = optimizerG
        self.get_noise = noise_generator
        self.device = opt.device
        self.one = torch.tensor(1, dtype=torch.float, device=self.device)
        self.clip_range = opt.weight_clip # 超参数
        self.use_gp = opt.use_gp
        self.gp_lambda = opt.gp_lambda
        self.logger = logger

        self.logger = Logger(opt)


    def updateD(self, images, step):

        images = images.to(self.device)
        batch_size = images.size(0)
        z = self.get_noise(batch_size)

        ## train the critic
        close_grad(self.G)
        self.optimizerD.zero_grad()
        out_true = self.D(images)
        out_true.mean().backward(self.one * -1)

        fake_images = self.G(z)
        out_fake = self.D(fake_images)
        out_fake.mean().backward(self.one, retain_graph = self.use_gp)

        if self.use_gp:
            gp = calculate_gradient_penalty(
                self.D, images, fake_images, self.gp_lambda, self.device)
            gp.backward()
            self.optimizerD.step()
            self.logger.writer.add_scalar("GP", gp.item(), global_step=step)
        else:
            self.optimizerD.step()
            weight_clip(self.D, self.clip_range)

        open_grad(self.G)
        
        # record 
        lossD = out_fake.mean().item() - out_true.mean().item()
        wasserstain_distance = -lossD

        self.logger.writer.add_scalar("lossD", lossD, global_step=step)
        self.logger.writer.add_scalar("Wasserstain Distance", wasserstain_distance, global_step=step)
        
        pass 


    def updateG(self, images, step):

        z = self.get_noise(images.size(0))
        close_grad(self.D)
        
        self.optimizerG.zero_grad()
        fake_images = self.G(z)
        out_fake = self.D(fake_images)
        out_fake.mean().backward(self.one * -1)
        self.optimizerG.step()

        open_grad(self.D)

        lossG = - out_fake.mean().item()
        self.logger.writer.add_scalar("lossG", lossG, global_step=step)
        self.logger.add_images("true", images)
        self.logger.add_images("fake", fake_images)

        return fake_images


class Updater:
    def __init__(self, opt, netG, netD, optimizerD, optimizerG, noise_generator, 
        logger):
        self.G = netG
        self.D = netD
        self.optimizerD = optimizerD
        self.optimizerG = optimizerG
        self.get_noise = noise_generator
        self.device = opt.device
        self.one = torch.tensor(1, dtype=torch.float, device=self.device)
        self.zero = torch.tensor(0, dtype=torch.float, device=self.device)
        self.criterion = nn.BCELoss()
        self.logger = logger

    def updateD(self, images, step):

        images = images.to(self.device)
        batch_size = images.size(0)
        z = self.get_noise(batch_size)
        label_true = self.one.expand((batch_size, 1))
        label_fake = self.zero.expand((batch_size, 1))

        ## train the discriminator
        close_grad(self.G)
        self.optimizerD.zero_grad()
        
        out_true = self.D(images)
        fake_images = self.G(z)
        out_fake = self.D(fake_images)
        lossD = self.criterion(out_true, label_true) + self.criterion(out_fake, label_fake)
        lossD.backward()

        self.optimizerD.step()
        open_grad(self.G)
        
        # record 
        lossD = lossD.item()
        self.logger.writer.add_scalar("lossD", lossD, global_step=step)


    def updateG(self, images, step):

        batch_size = images.size(0)
        z = self.get_noise(batch_size)
        label_fake = self.zero.expand((batch_size, 1))

        close_grad(self.D)
        self.optimizerG.zero_grad()

        fake_images = self.G(z)
        out_fake = self.D(fake_images)
        lossG = self.criterion(out_fake, label_fake)
        lossG.backward()

        self.optimizerG.step()
        open_grad(self.D)

        # record
        lossG = lossG.item()
        self.logger.writer.add_scalar("lossG", lossG, global_step=step)

        return fake_images

class LSUpdater:
    def __init__(self, opt):
        pass 

    def updateD():
        pass 

    def updateG():
        pass