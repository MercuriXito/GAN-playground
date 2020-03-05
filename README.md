# GANs

Project to train several GANs with pytorch on different Datasets.

Models Available:

- [x] Vanilla GAN (with MLP)
- [x] DCGAN
- [x] WGAN
- [x] WGAN-GP
- [ ] LSGAN

Dataset Avaiable:

- [ ] MNIST
- [ ] CIFAR-10
- [ ] [AnimeFaces](http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/)


## Training Record

| model | lr_G | lr_D | epochs | methods | dataset |
| -- | -- | -- | -- | -- | -- |
| MLP | 0.0002 | 0.0002 | 40 | Adam betas = (0.5, 0.999) | MNIST |
| DCGAN | 0.0002 | 0.0002 | 200 | Adam betas = (0.5, 0.999) | AnimeFaces |
| WGAN | 5e-5 | 5e-5 | 50 | RMSprop | AnimeFaces |
| WGAN-GP | 1e-4 | 1e-4 | 200 | Adam betas = (0,0.9), $\lambda=10$ | AnimeFaces |


## Reference

- DCGAN: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks]()
- WGAN: [Wasserstain Generative Adversarial Networks]()
- WGAN-GP: [Improved Training of Wasserstain GANs]()
- LSGAN: [Least Squares Generative Adversarial Networks]()