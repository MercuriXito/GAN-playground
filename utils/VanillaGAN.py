#-*-coding:utf-8-*-

"""
    @file:			vanilla_train.py
    @autor:			Victor Chen
    @description:
        vanilla training process 原paper的训练方法
"""

import torch
import torch.optim as optim
import numpy as np

from utils.visulize import save_batch

class GANTrainer:
    """ Vanilla Training process suggested in original paper
    """

    def __init__(self, epochs, lr_G, lr_D, folder_name):
        
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.epochs = epochs
        self.generate_batch = 128
        self.image_save_path = "saved/images/{}/".format(folder_name)
        self.model_save_path = "saved/models/"


    def train(self, gen, dis, dataloader, noise_generator, device=None):

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        optimizor_G = optim.Adam(gen.parameters(), lr = self.lr_G, betas = (0.5, 0.999))
        optimizor_D = optim.Adam(dis.parameters(), lr = self.lr_D, betas = (0.5, 0.999)) 

        cost = torch.nn.BCELoss()
        print("Start Training, using device:{}".format(device.type))
        for epoch in range(self.epochs):
            print("Epoch:{}/{}".format(epoch + 1, self.epochs))

            epoch_loss_G = 0.0
            epoch_loss_D = 0.0
            for i, (images, _) in enumerate(dataloader):
                images = images.to(device)
                batch_size = images.size(0)

                ##########
                # train discriminator
                ##########
                optimizor_D.zero_grad()
                out_true = dis(images)

                z = noise_generator(self.generate_batch).to(device)
                fake_images = gen(z)
                out_fake = dis(fake_images)

                labels_true = torch.ones(batch_size).to(device)
                labels_fake = torch.zeros(self.generate_batch).to(device)
                cost_D = cost(out_true, labels_true) + cost(out_fake, labels_fake)

                cost_D.backward()
                optimizor_D.step()
                epoch_loss_D += cost_D.item()

                ##########
                # train generator
                ##########
                optimizor_G.zero_grad()
                
                fake_images = gen(z)
                out_fake = dis(fake_images)
                labels_fake_true = torch.ones(self.generate_batch).to(device)
                cost_G = cost(out_fake, labels_fake_true )

                cost_G.backward()
                optimizor_G.step()
                epoch_loss_G += cost_G.item()

                with torch.no_grad():
                    print("[ loss_G: %.6f ] - [ loss_D: %.6f ]" %( epoch_loss_G/(i+1), epoch_loss_D /(i+1) )) 
                    print("[ best_fake_poss: %.4f%% ] [ worst_fake_poss: %.4f%% ] [avg_fake_poss: %.4f%% ]" %(
                        torch.max(out_fake).item(), torch.min(out_fake).item(), torch.mean(out_fake).item()
                    ))
                
                if i % 30 == 0:
                    save_batch(images, "{}{}_true.png".format(self.image_save_path, epoch))
                    save_batch(fake_images, "{}{}.png".format(self.image_save_path, epoch))

        print("[ loss_G: %.6f ] - [ loss_D: %.6f ]" %( epoch_loss_G/(i+1), epoch_loss_D /(i+1) )) 
        save_batch(images, "{}{}_true.png".format(self.image_save_path, epoch))
        save_batch(fake_images, "{}{}.png".format(self.image_save_path, epoch))

        # save models
        save_gen_name = "generator_{}_{}.pth".format(gen.__class__.__name__, epoch + 1)
        save_dis_name = "discriminator_{}_{}.pth".format(dis.__class__.__name__, epoch + 1)
        with open(self.model_save_path + save_gen_name, "wb") as f:
            torch.save(gen.state_dict(), f)

        with open(self.model_save_path + save_dis_name, "wb") as f:
            torch.save(dis.state_dict(), f)
    
        pass
