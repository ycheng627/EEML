import numpy as np
import torch
import torch.nn as nn

cuda_num = input("Input cude number: ")
model_num = input("Input model Number: ")
contin = input("Continue from last checkpoint?[y/n] ")

dropout = 0.2
lr = 1e-3

accuracy_list = [0]

train_config = {
    "data_dir": "/tmp2/b07902084/EEML/hw4/Dataset",
    "save_path": "/tmp2/b07902084/EEML/hw4/model-{}.ckpt".format(model_num),
    "batch_size": 32,
    "n_workers": 8,
    "valid_steps": 2000,
    "warmup_steps": 2000,
    "save_steps": 10000,
    "total_steps": 1000000,
}

if contin == 'y':
    train_config["warmup_steps"] = 0

test_config = {
    "data_dir": "/tmp2/b07902084/EEML/hw4/Dataset",
    "model_path": "/tmp2/b07902084/EEML/hw4/model-{}.ckpt".format(model_num),
    "output_path": "./output-{}-{}.csv".format(model_num, 0),
}