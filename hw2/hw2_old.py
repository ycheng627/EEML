import numpy as np
import sys

GPU_num = input("Input GPU num (enter for 0):")

print('Loading data ...')

data_root='/tmp2/b07902084/EEML/hw2/timit_11/'
train = np.load(data_root + 'train_11.npy', mmap_mode="r")
print("load train")
train_label = np.load(data_root + 'train_label_11.npy', mmap_mode="r")
print("load label")
test = np.load(data_root + 'test_11.npy', mmap_mode="r")
print("load test")

train = train.reshape(-1, 11, 39)
test = test.reshape(-1, 11, 39)
train = np.swapaxes(train,1,2)
test = np.swapaxes(test,1,2)
# train_label = train_label.squeeze()


import torch
from torch.utils.data import Dataset

class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


VAL_RATIO = 0.2

# percent = int(train.shape[0] * (1 - VAL_RATIO))
percent = int(train.shape[0] * (VAL_RATIO))
train_x, train_y, val_x, val_y = train[percent:], train_label[percent:], train[:percent], train_label[:percent]
print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))
train_size = train_x.shape[0]
val_size = val_x.shape[0]

BATCH_SIZE = 4096

from torch.utils.data import DataLoader

train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1) #only shuffle the training data
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

import gc

del train, train_label, train_x, train_y, val_x, val_y
gc.collect()

import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(39, 128, 6),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.Dropout(p=0.6),
            nn.ReLU(),

            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.7),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.6),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.4),
            nn.ReLU(),

            nn.Linear(512, 39)
        )

    def forward(self, x):
#         print("initial shape: ", x.shape)
        x = self.conv(x)
#         print("after convolution: ", x.shape)
        x = torch.flatten(x, start_dim = 1)
#         print("after flatten: ", x.shape)
        x = self.fc(x)
#         print("final: ", x.shape)
        return x


#check device
my_cuda = "cuda:0"
if GPU_num:
    my_cuda = "cuda:{}".format(GPU_num)

def get_device():
  return my_cuda if torch.cuda.is_available() else 'cpu'

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# fix rand1m seed for reproducibility
same_seeds(1)

# get device 
device = get_device()
print(f'DEVICE: {device}')

# training parameters
num_epoch = 5000               # number of training epoch
learning_rate = 0.00006       # learning rate
weight_decay = 0.00001
early_stop = 200

# the path where checkpoint saved
model_path = '/tmp2/b07902084/EEML/hw2/model-{}.ckpt'.format(GPU_num)

# create model, define a loss function, and optimizer
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

print(model)
print("Learning Rate: {:3.6f}".format(learning_rate))
print("Weight Decay: {:3.6f}".format(weight_decay))

log = open("/tmp2/b07902084/EEML/hw2/log-{}.txt".format(GPU_num), "w", buffering=1) # No buffer flush immediately
print("log path: " + "/tmp2/b07902084/EEML/hw2/log-{}.txt".format(GPU_num))

# start training
from tqdm import *
best_acc = 0.0
best_acc_index = 0

with trange(num_epoch, position=0, desc='Acc: ', bar_format='{desc:20}{percentage:3.0f}%|{bar:50}{r_bar}') as outter_range:
    for epoch in outter_range:
        leave = epoch == len(outter_range) - 1
# for epoch in tqdm(range(num_epoch)):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        if epoch - best_acc_index > early_stop:
            break

        # training
        model.train() # set the model to training mode
        for i, data in enumerate(tqdm(train_loader, leave=leave, position=1, desc='Iters: ', bar_format='{desc:20}{percentage:3.0f}%|{bar:50}{r_bar}')):
            
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() 
            outputs = model(inputs) 
            


            batch_loss = criterion(outputs, labels)
    #             print("loss: ", batch_loss)
            _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
            batch_loss.backward() 
            optimizer.step() 

            train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
            train_loss += batch_loss.item()

        # validation
        if len(val_set) > 0:
            model.eval() # set the model to evaluation mode
            with torch.no_grad():
                for i, data in enumerate((val_loader)):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
    #                 print("inputs: ", inputs.shape)
    #                 print("Labels: ", labels.shape)
                    outputs = model(inputs)
    #                 print("output: ", outputs.shape)
                    batch_loss = criterion(outputs, labels) 
                    _, val_pred = torch.max(outputs, 1) 
                
                    val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                    val_loss += batch_loss.item()

                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                    epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
                ), file=log)

                # if the model improves, save a checkpoint at this epoch
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_acc_index = epoch
                    torch.save(model.state_dict(), model_path)
                    print('saving model with acc {:.3f}'.format(best_acc/len(val_set)), file=log)
        else:
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader)
            ), file=sys.stderr)
        outter_range.set_description("Cur Acc: {:3.6f}".format(val_acc/len(val_set)))

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), "/home/student/07/b07902084/EEML/hw2/model-{}.ckpt".format(best_acc))
    print('saving model at last epoch')


# create testing dataset
test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# create model and load weights from checkpoint
model = Classifier().to(device)
model.load_state_dict(torch.load(model_path))

predict = []
model.eval() # set the model to evaluation mode
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability

        for y in test_pred.cpu().numpy():
            predict.append(y)

with open('prediction-{}.csv'.format(best_acc/len(val_set)), 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))
print("finish prediction")
