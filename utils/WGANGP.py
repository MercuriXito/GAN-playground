#-*-coding:utf-8-*-

"""
    @file:			wgan_train.py
    @autor:			Victor Chen
    @description:
        training process suggested in paper WGAN-GP
"""

import torch
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import math

from tensorboardX import SummaryWriter

from utils.visulize import save_batch_images
from utils.record import record_gradient

def calc_gradient_penalty(netD, real_data, fake_data, batch_size,
             img_channels, img_size, device, gp_lambda):
    """ modified from https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
    """
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, img_channels,  img_size, img_size)
    alpha = alpha.to(device)
    
    fake_data = fake_data.view(batch_size, img_channels,  img_size, img_size)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty


class WGANGPTrainer:
    """ WGAN Training process
    """

    def __init__(self, epochs, lr_G, lr_D, save_images_name, record_name = "WGANGP"):

        self.lr_G = lr_G
        self.lr_D = lr_D
        self.epochs = epochs
        self.train_critic_times = 5 # times of training critic
        self.generate_batch = 32
        self.image_save_path = "saved/images/{}/".format(save_images_name)
        self.model_save_path = "saved/models/"
        self.writer = SummaryWriter(logdir="saved/checked/"+record_name+"/")
        self.gradient_penalty_lambda = 10


    def train(self, G, critic, dataloader, noise_generator, device=None):

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        optimizor_G = optim.Adam(G.parameters(), lr = self.lr_G, betas=(0,0.9))
        optimizor_D = optim.Adam(critic.parameters(), lr = self.lr_D, betas=(0,0.9))

        print("Start Training, using device:{}".format(device.type))
        
        train_generator_times = 0.0
        for epoch in range(self.epochs):
            print("Epoch:{}/{}".format(epoch + 1, self.epochs))

            epoch_loss_G = 0.0
            epoch_loss_D = 0.0

            loader = iter(dataloader)
            dataset_size = len(dataloader.dataset)
            i = 0

            while i < dataset_size:
                avg_loss_D = 0.0
                Wasserstain_D = 0.0
                for j, (images, _) in enumerate(loader):

                    images = images.to(device)
                    i += images.size(0)
                    img_channels, img_size = images.size(1), images.size(-1)

                    ############
                    # train the discriminator
                    ############

                    optimizor_D.zero_grad()
                    out_true = critic(images)

                    z = noise_generator(images.size(0)).to(device)
                    fakeimages = G(z)
                    out_fake = critic(fakeimages)

                    loss_D = out_fake.mean() - out_true.mean()
                    loss_D.backward(retain_graph = True)

                    # ------ train with gradient penalty
                    gp = calc_gradient_penalty(critic, images, fakeimages, images.size(0), 
                        img_channels, img_size, device , self.gradient_penalty_lambda)
                    gp.backward()

                    optimizor_D.step()

                    avg_loss_D += loss_D.item() + gp.item()
                    Wasserstain_D += - loss_D.item()
                    if j == self.train_critic_times - 1:
                        break

                avg_loss_D = avg_loss_D / self.train_critic_times
                Wasserstain_D = Wasserstain_D / self.train_critic_times

                #######
                # train the generator
                optimizor_G.zero_grad()

                fakeimages = G(z)
                out_fake = critic(fakeimages)

                loss_G = - out_fake.mean()
                loss_G.backward()
                optimizor_G.step()
                avg_loss_G = loss_G.item()

                print("[ loss_G: %.6f ] - [ loss_D: %.6f ] - [ Wasserstain_D: %.6f ]" 
                        %(avg_loss_G, avg_loss_D, Wasserstain_D))

                global_step = train_generator_times
                train_generator_times += 1
                self.writer.add_scalar("loss_D", avg_loss_D, global_step)
                self.writer.add_scalar("loss_G", avg_loss_G, global_step)
                self.writer.add_scalar("Wasserstain dis", Wasserstain_D, global_step)

                if i % 100 == 0:
                    save_batch_images(images, "{}_true".format(epoch), root=self.image_save_path)
                    save_batch_images(fakeimages, "{}".format(epoch), root=self.image_save_path)
                    

        print("[ loss_G: %.6f ] - [ loss_D: %.6f ]" %( epoch_loss_G/(i+1), epoch_loss_D /(i+1) )) 
        save_batch_images(fakeimages, "{}".format(epoch), root=self.image_save_path)

        # save models
        save_gen_name = "generator_{}_{}.pth".format(G.__class__.__name__, epoch + 1)
        save_critic_name = "critic_{}_{}.pth".format(critic.__class__.__name__, epoch + 1)
        with open(self.model_save_path + save_gen_name, "wb") as f:
            torch.save(G.state_dict(), f)

        with open(self.model_save_path + save_critic_name, "wb") as f:
            torch.save(critic.state_dict(), f)
    
        pass

