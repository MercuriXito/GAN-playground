# GANs

Implementation of serveral GANs with pytorch:

- [x] Vanilla GAN (with MLP)
- [x] DCGAN
- [ ] WGAN
- [ ] WGAN-GP
- [ ] SA-GAN (Attention)
- [ ] cGAN
- [ ] pix2pix

## Training Condition

| model | lr_G | lr_D | epochs | methods | dataset |
| -- | -- | -- | -- | -- | -- |
| MLP | 0.0002 | 0.0002 | 40 | Adam betas = (0.5, 0.999) | MNIST |
| DCGAN | 0.0002 | 0.0002 | 40 | Adam betas = (0.5, 0.999) | AnimeFaces |
| WGAN | 5e-5 | 5e-5 | 50 | RMSprop | AnimeFaces |
| WGAN-GP |

## Record

- It's a little strange to fail to train the WGAN(Weight-Clipping ver), the generator worked at first and learned some features like shape, but after a while, the generator seemingly could not improve anymore, **the generated images often lack details and structure**, and **the Wasserstain Distance preserve at the same level**. (2019.12.19)
  
   > An example generated images after training WGAN after 48 epoch.
   ![2019-12-19-failed-WGAN](docs/imgs/20191219_WGAN_48epoch.png)

- It seems that the problem also occur when training WGAN-GP. Then Wasserstain loss is around 2.

   > Another example generated images after training WGAN-GP for 39 epochs.
   ![2019-12-19-failed-WGANGP](docs/imgs/20191219-WGANGP_39epochs.png)