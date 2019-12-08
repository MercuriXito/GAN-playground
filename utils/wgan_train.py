#-*-coding:utf-8-*-

"""
    @file:			wgan_train.py
    @autor:			Victor Chen
    @description:
        training process suggested in paper WGAN
        在WGAN论文中提到的训练方法：
            * 修改 loss 的计算方法：不加 log, critic 输出层不加 sigmoid
            * 对 critic 做 weight_clipping
            * 不使用集于动量的优化算法
"""

import torch
import torch.optim as optim
import numpy as np

from utils.visulize import save_batch_images


def weight_clipping(model, clipped_range = 0.01):
    for param in model.parameters():
        param.data.clamp_(-clipped_range, clipped_range)

class AdvTrainer:
    """ Vanilla Training process suggested in original paper
    """

    def __init__(self, epochs, lr_G, lr_D, save_images_name):
        
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.epochs = epochs
        self.train_critic_times = 10 # times of training 
        self.generate_batch = 32
        self.image_save_path = "saved/images/{}/".format(save_images_name)
        self.model_save_path = "saved/models/"


    def train(self, G, critic, dataloader, noise_generator, device=None):

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        optimizor_G = optim.RMSprop(G.parameters(), lr = self.lr_G) # using RMSprop
        optimizor_D = optim.RMSprop(critic.parameters(), lr = self.lr_G)

        print("Start Training, using device:{}".format(device.type))
        for epoch in range(self.epochs):
            print("Epoch:{}/{}".format(epoch + 1, self.epochs))

            epoch_loss_G = 0.0
            epoch_loss_D = 0.0

            for i, (images, _) in enumerate(dataloader):
                images = images.to(device)

                ##########
                # train critic
                ##########
                optimizor_D.zero_grad()
                out_true = critic(images)

                z = noise_generator(self.generate_batch).to(device)
                fake_images = G(z)
                out_fake = critic(fake_images)

                cost_D = out_fake.mean() - out_true.mean() # loss suggested in WGAN
                cost_D.backward()
                optimizor_D.step()
                epoch_loss_D += cost_D.item()

                weight_clipping(critic, 0.1) # weight clipping 
                ##########
                # train generator
                ##########
                optimizor_G.zero_grad()
                
                fake_images = G(z)
                out_fake = critic(fake_images)

                cost_G = - out_fake.mean() # loss suggested in WGAN
                cost_G.backward()
                optimizor_G.step()
                epoch_loss_G += cost_G.item()

                with torch.no_grad():
                    print("[ loss_G: %.6f ] - [ loss_D: %.6f ]" %( epoch_loss_G/(i+1), epoch_loss_D /(i+1) )) 
                
                if i % 100 == 0:
                    save_batch_images(images, "{}_true".format(epoch), root=self.image_save_path)
                    save_batch_images(fake_images, "{}".format(epoch), root=self.image_save_path)

        print("[ loss_G: %.6f ] - [ loss_D: %.6f ]" %( epoch_loss_G/(i+1), epoch_loss_D /(i+1) )) 
        save_batch_images(fake_images, "{}".format(epoch), root=self.image_save_path)

        # save models
        save_gen_name = "generator_{}_{}.pth".format(G.__class__.__name__, epoch + 1)
        save_critic_name = "critic_{}_{}.pth".format(critic.__class__.__name__, epoch + 1)
        with open(self.model_save_path + save_gen_name, "wb") as f:
            torch.save(G.state_dict(), f)

        with open(self.model_save_path + save_critic_name, "wb") as f:
            torch.save(critic.state_dict(), f)
    
        pass

