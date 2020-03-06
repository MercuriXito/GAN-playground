import argparse
import os, sys
import json
import torch

"""
设计： 使用命令行的形式可以在不同的训练集上训练GAN，并可以调整参数，参数包括：

参数也不用给太多，能跑个效果就行。

+ models: 选择什么模型
+ dataset: 选择什么数据集

生成参数：

+ 生成的图片的大小: 32x32

训练参数：

+ lr (G,D)
+ epochs (G,D)
+ optimizer (SGD, Adam) for (G,D)
+ 两种优化器的超参数
+ train_interval (G,D). G 和 D 的训练比例
+ batch_size 
+ noise_size
+ device

持久化参数:

+ 隔多少个 epoch 保存一次模型
+ 保存路径
+ dataset 路径
+ 一个epoch内保存几次模型

"""

allowed_model = ["wgan","wgangp","lsgan","gan","dcgan"]
allowed_dataset = ["animefaces","mnist","cifar10"]
allowed_image_size = [64, 32]

def process_config(config):

    if config.G_train_interval == -1 or config.D_train_interval == -1:
        if "wgan" in config.model:
            config.G_train_interval = 5
            config.D_train_interval = 1
        else:
            config.G_train_interval = 1
            config.D_train_interval = 1

    if config.save_model_interval == -1:
        config.save_model_interval = config.epochs

    # select device
    if not torch.cuda.is_available() and config.device == "cuda":
            config.device = "cpu"
            print("Cuda not available, using cpu.")

    # set hyper-parameters
    use_gp = True if config.model == "wgangp" else False
    setattr(config, "use_gp", use_gp)
    
    return config


def train_config_parser():

    parser = argparse.ArgumentParser()
    # 训练参数
    parser.add_argument("model", choices=allowed_model)
    parser.add_argument("dataset", choices=allowed_dataset)

    # 生成参数
    parser.add_argument("-is", "--image-size", choices=allowed_image_size, default = 32,  type=int)
    parser.add_argument("--save-images-nrow", default=16, type=int)

    # 训练参数
    parser.add_argument("-G_lr", default=1e-4, type=float)
    parser.add_argument("-D_lr", default=1e-4, type=float)
    parser.add_argument("-e", "--epochs", default=50, type=int)
    parser.add_argument("-G_optim", "--G_optimizer", choices=["adam","sgd"], default="adam", type=str)
    parser.add_argument("-D_optim", "--D_optimizer", choices=["adam","sgd"], default="adam", type=str)
    parser.add_argument("--adam-betas", default=((0.5,0.99)), type=tuple)
    parser.add_argument("--sgd-momentum", default=0.9, type=float)
    parser.add_argument("-Gtr", "--G-train-interval", default=-1) # if not chosen, will be chosen according to the chosen model
    parser.add_argument("-Dtr", "--D-train-interval", default=-1)
    parser.add_argument("--batch-size", default=64, type = int)
    parser.add_argument("--noise-size", default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", default=16, type = int)
    parser.add_argument("--shuffle", action="store_false")

    # 持久化参数
    parser.add_argument("--save-root", default = "save/", type=str)
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--save-model-interval", default=-1, type=int) # epoch interval to save model
    parser.add_argument("--save-num-image", default=5, type=int) # number of image to save in one epoch
    # parser.add_argument("--only-show", action="store_true") # switch to show mode, do not save the image

    # WGAN 的超参数
    # Weight Clipping 的 范围 
    parser.add_argument("--weight_clip", default=0.02, type=float)

    # WGAN-GP 的超参数
    # Gradient Penalty 的 lambda 
    parser.add_argument("--gp-lambda", default=10, type=float)

    # LS-GAN 的超参数
    # a, b 的取值

    # 推断时的参数

    config = parser.parse_args()
    config = process_config(config)
    return config



"""
推断的时候的参数：

全局参数：
+ 可以读取保存时的完整目录，并从模型中的 json 中恢复，保存的目录的模型的参数，图片的参数：

除此之外有关的训练参数都不用管：

这样剩余的参数有：

+ --save-images-nrow 也要可不要，可以调整

+ --batch-size: 可以调整

+ --device： 可以调整

引入的其他参数：
+ --read-save-root: 读取的 root，可能是唯一需要设置的参数
+ --show-transition： 画 transition 的图片,
+ --transition-number： transition 中间的多少。

"""

def process_infer_config(config):

    # select device
    if not torch.cuda.is_available() and config.device == "cuda":
            config.device = "cpu"
            print("Cuda not available, using cpu.")

    # set hyper-parameters
    use_gp = True if config.model == "wgangp" else False
    setattr(config, "use_gp", use_gp)
    
    return config


def read_options(config):
    """ Read parameters in saved pre-trained config specified by config.root
    Typical save folder is as follows:
    
        /root 
            /models
                /model_G_dataset.pth
                /model_D_dataset.pth
                /config_model_dataset.json # config file
            /images
                /...
            event-... # for tensorboard
    """

    root = config.root
    models_root = root + "models/"

    jsonfiles = ""
    for file in os.listdir(models_root):
        if file.split(".")[-1] == 'json' and file[:6] == "config":
            jsonfiles = file

    # print(jsonfiles)
    with open(models_root + jsonfiles, "r") as f:
        pre_config = json.loads(f.read())

    named = jsonfiles.split(".")[0]
    assert pre_config["model"] == named.split("_")[1], "Wrong Models"
    assert pre_config["dataset"] == named.split("_")[2], "Wrong Dataset"

    dict_config = dict(config._get_kwargs())
    for key, val in pre_config.items():
        if key in dict_config.keys():
            continue
        setattr(config, key, val)

    return config


def inference_config_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--read-save-root", type=str, dest="root", required=True)
    parser.add_argument("--gen-batch", type=int, default=32)
    parser.add_argument("--gen-images-nrow", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--show-transition", action="store_true")
    parser.add_argument("--transition-number", type=int, default=8)

    config = parser.parse_args()
    config = read_options(config)
    config = process_infer_config(config)
    return config


if __name__ == "__main__":
    
    # opt = train_config_parser()
    # print(dict(opt._get_kwargs()))
    # print(opt)
    opt = inference_config_parser()
    print("Using parameters:")
    for name, value in dict(opt._get_kwargs()).items():
        print("%25s:\t%s" %(name,value))