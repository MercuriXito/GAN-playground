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

from utils.visulize import save_batch_images
from utils.record import record_gradient

def calc_gradient_penalty(netD, real_data, fake_data, batch_size, dim, device, gp_lambda):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, dim, dim)
    alpha = alpha.to(device)
    
    fake_data = fake_data.view(batch_size, 3, dim, dim)
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

    def __init__(self, epochs, lr_G, lr_D, save_images_name):
        
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.epochs = epochs
        self.train_critic_times = 5 # times of training critic
        self.generate_batch = 32
        self.weight_range = 0.01
        self.image_save_path = "saved/images/{}/".format(save_images_name)
        self.model_save_path = "saved/models/"


    def train(self, G, critic, dataloader, noise_generator, device=None):

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # optimizor_G = optim.RMSprop(G.parameters(), lr = self.lr_G) # using RMSprop
        # optimizor_D = optim.RMSprop(critic.parameters(), lr = self.lr_G)

        optimizor_G = optim.Adam(G.parameters(), lr = self.lr_G)
        optimizor_D = optim.Adam(critic.parameters(), lr = self.lr_D)

        print("Start Training, using device:{}".format(device.type))
        
        for epoch in range(self.epochs):
            print("Epoch:{}/{}".format(epoch + 1, self.epochs))

            epoch_loss_G = 0.0
            epoch_loss_D = 0.0

            loader = iter(dataloader)
            dataset_size = len(dataloader.dataset)
            i = 0

            while i < dataset_size:
                avg_loss_D = 0.0
                for j, (images, _) in enumerate(loader):

                    images = images.to(device)
                    i += images.size(0)

                    ############
                    # train the discriminator
                    ############

                    optimizor_D.zero_grad()
                    out_true = critic(images)
                    out_true.backward()

                    z = noise_generator(images.size(0)).to(device)
                    fakeimages = G(z)
                    out_fake = critic(fakeimages)

                    loss_D = out_fake.mean() - out_true.mean()
                    loss_D.backward()

                    # ------ train with gradient penalty
                    gp = calc_gradient_penalty(critic, images, fakeimages, images.size(0), 64, device , 10)
                    gp.backward()

                    optimizor_D.step()

                    avg_loss_D += loss_D.item()
                    if j == self.train_critic_times - 1:
                        break

                avg_loss_D = avg_loss_D / self.train_critic_times

                #######
                # train the generator
                optimizor_G.zero_grad()

                fakeimages = G(z)
                out_fake = critic(fakeimages)

                loss_G = - out_fake.mean()
                loss_G.backward()
                optimizor_G.step()
                avg_loss_G = loss_G.item()

                print("[ loss_G: %.6f ] - [ loss_D: %.6f ]" %(avg_loss_G, avg_loss_D))

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

