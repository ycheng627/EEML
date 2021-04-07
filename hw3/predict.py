cuda_num = input("Enter cuda number: ")

# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder
from tqdm import tqdm

import my_model
import my_transforms
import my_configs
import pseudolabel


train_tfm = my_transforms.get_train_tfm()
test_tfm = my_transforms.get_test_tfm()

batch_size = my_configs.batch_size

doc_root = "/tmp2/b07902084/EEML/hw3/"
train_set = DatasetFolder(doc_root + "food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
valid_set = DatasetFolder(doc_root + "food-11/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
unlabeled_set = DatasetFolder(doc_root + "food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
test_set = DatasetFolder(doc_root + "food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# "cuda" only when GPUs are available.
device = "cuda:{}".format(cuda_num) if torch.cuda.is_available() else "cpu"
# device = "cpu"

model = my_model.model.to(device)
model.device = device

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=my_configs.lr, weight_decay=my_configs.wd)
n_epochs = my_configs.n_epochs
do_semi = my_configs.do_semi


model_path = '/tmp2/b07902084/EEML/hw3/model-C-4.ckpt'.format(cuda_num)
best_acc = 0


model = my_model.model.to(device)
model.load_state_dict(torch.load(model_path))


# Make sure the model is in eval mode.
# Some modules like Dropout or BatchNorm affect if the model is in training mode.
model.eval()

# Initialize a list to store the predictions.
predictions = []

# Iterate the testing set by batches.
for batch in tqdm(test_loader):
    # A batch consists of image data and corresponding labels.
    # But here the variable "labels" is useless since we do not have the ground-truth.
    # If printing out the labels, you will find that it is always 0.
    # This is because the wrapper (DatasetFolder) returns images and labels for each batch,
    # so we have to create fake labels to make it work normally.
    imgs, labels = batch

    # We don't need gradient in testing, and we don't even have labels to compute loss.
    # Using torch.no_grad() accelerates the forward process.
    with torch.no_grad():
        logits = model(imgs.to(device))

    # Take the class with greatest logit as prediction and record it.
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

# Save predictions into the file.
with open("predict-{}.csv".format(best_acc), "w") as f:

    # The first row must be "Id, Category"
    f.write("Id,Category\n")

    # For the rest of the rows, each image id corresponds to a predicted class.
    for i, pred in  enumerate(predictions):
         f.write(f"{i},{pred}\n")
