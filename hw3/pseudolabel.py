import numpy as np
import torch
import torch.nn as nn
import my_configs
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from tqdm import tqdm

from torch.utils.data import Dataset

class PseudoDataset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, labels, indices, device):
        self.dataset = Subset(dataset, indices)
        self.targets = labels
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)

def get_pseudo_labels(dataset, model, device, threshold=0.75):

    data_loader = DataLoader(dataset, batch_size=my_configs.batch_size, shuffle=False, num_workers=8)

    model.eval()
    softmax = nn.Softmax(dim=-1)

    features = 0
    indices = []
    labels = []
    for iteration, batch in enumerate(tqdm(data_loader)):
        # if iteration > 20:
        #     break
        img, _ = batch
        cnt = my_configs.batch_size * iteration

        with torch.no_grad():
            logits = model(img.to(device))

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)
        m = torch.max(probs, dim=1)
        p = m[0]
        l = m[1]

        for i in range(len(m)):
            if p[i] > threshold:
                indices.append(cnt + i)
                labels.append(l[i].item())

        # if iteration == 0:
        #     labels = l
        #     indices = p
        # else:
        #     labels = torch.cat((labels, this_labels), 0)
        #     indices += this_index
    my_dataset = PseudoDataset(dataset, labels, indices, device)
    # # Turn off the eval mode.
    model.train()
    return my_dataset