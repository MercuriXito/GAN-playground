# GANs

Implementation of serveral GANs with pytorch:

- [x] Vanilla GAN (with MLP)
- [ ] DCGAN
- [ ] WGAN
- [ ] WGAN-GP
- [ ] SaGAN

## Training Condition

| model | lr_G | lr_D | epochs | methods | dataset |
| -- | -- | -- | -- | -- | -- |
| MLP | 0.0002 | 0.0002 | 40 | Adam betas = (0.5, 0.999) | MNIST |
| DCGAN | 0.0002 | 0.0002 | 40 | Adam betas = (0.5, 0.999) |   |
