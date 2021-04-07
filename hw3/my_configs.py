import numpy as np
import torch
import torch.nn as nn


# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
batch_size = 32


def get_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)


# The number of training epochs.
n_epochs = 1000

# Whether to do semi-supervised learning.
do_semi = False

lr = 0.001
wd = 1e-5

semi_start = 0.7
semi_thresh = 0.95
