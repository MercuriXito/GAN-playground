#-*-coding:utf-8-*-

"""
    @file:			wgan_train.py
    @autor:			Victor Chen
    @description:
        training process suggested in paper WGAN
        在WGAN论文中提到的训练方法：
            * 修改 loss 的计算方法：不加 log, critic 输出层不加 sigmoid
            * 对 critic 做 weight_clipping
            * 不使用基于动量的优化算法
"""

import torch
import torch.optim as optim
import numpy as np
import math

from utils.visulize import save_batch
from utils.record import record_gradient

from tensorboardX import SummaryWriter

def weight_clipping(model, clipped_range = 0.01):
    for param in model.parameters():
        param.data.clamp_(-clipped_range, clipped_range)

class WGANTrainer:
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
        self.writer = SummaryWriter(logdir="saved/checked/model/", comment="WGAN")


    def train(self, G, critic, dataloader, noise_generator, device=None):

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        optimizor_G = optim.RMSprop(G.parameters(), lr = self.lr_G) # using RMSprop
        optimizor_D = optim.RMSprop(critic.parameters(), lr = self.lr_G)

        one = torch.tensor([1], dtype=torch.float32, device=device)

        print("Start Training, using device:{}".format(device.type))
        
        means = dataloader.dataset.mean
        stds = dataloader.dataset.std
        for epoch in range(self.epochs):
            print("Epoch:{}/{}".format(epoch + 1, self.epochs))

            epoch_loss_G = 0.0
            epoch_loss_D = 0.0

            loader = iter(dataloader)
            dataset_size = len(dataloader.dataset)
            batch_size = dataloader.batch_size
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
                    out_true.mean().backward(one * -1)

                    z = noise_generator(self.generate_batch).to(device)
                    fake_images = G(z)
                    out_fake = critic(fake_images)
                    out_fake.mean().backward(one)

                    plossD = out_fake.mean() - out_true.mean()
                    
                    optimizor_D.step()
                    avg_loss_D += plossD.item()
                    weight_clipping(critic, self.weight_range)

                    if j == self.train_critic_times - 1:
                        break

                avg_loss_D = avg_loss_D / self.train_critic_times

                #######
                # train the generator
                optimizor_G.zero_grad()

                fake_images = G(z)
                out_fake = critic(fake_images)
                out_fake.mean().backward(one * -1)

                avg_loss_G = - out_fake.mean().item()
                optimizor_G.step()

                print("[ loss_G: %.6f ] - [ loss_D: %.6f ] - [ Wasserstain_D: %.6f ]" 
                        %(avg_loss_G, avg_loss_D, -avg_loss_D))

                global_step = math.ceil( epoch * dataset_size + i ) / batch_size
                self.writer.add_scalar("train D loss", avg_loss_D, global_step)
                self.writer.add_scalar("train G loss", avg_loss_G, global_step)
                self.writer.add_scalar("Wasserstrain dis", - avg_loss_D, global_step)

                if i % 100 == 0:
                    save_batch(images, "{}{}_true.png".format(self.image_save_path, epoch), mean=means, std=stds)
                    save_batch(fake_images, "{}{}.png".format(self.image_save_path, epoch), mean=means, std=stds)
                    

        print("[ loss_G: %.6f ] - [ loss_D: %.6f ]" %( epoch_loss_G/(i+1), epoch_loss_D /(i+1) )) 
        save_batch(images, "{}{}_true.png".format(self.image_save_path, epoch), mean=means, std=stds)
        save_batch(fake_images, "{}{}.png".format(self.image_save_path, epoch), mean=means, std=stds)
 
        # save models
        save_gen_name = "generator_{}_{}.pth".format(G.__class__.__name__, epoch + 1)
        save_critic_name = "critic_{}_{}.pth".format(critic.__class__.__name__, epoch + 1)
        with open(self.model_save_path + save_gen_name, "wb") as f:
            torch.save(G.state_dict(), f)

        with open(self.model_save_path + save_critic_name, "wb") as f:
            torch.save(critic.state_dict(), f)
    
        pass

