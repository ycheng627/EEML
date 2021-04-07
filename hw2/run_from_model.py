import numpy as np
import sys
from tqdm import tqdm

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

print(test.shape)



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


BATCH_SIZE = 4096

from torch.utils.data import DataLoader


import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(39, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Dropout(p=0.5)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 11, 4096),
            nn.BatchNorm1d(4096),
            nn.Dropout(p=0.6),
            nn.ReLU(),

            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.6),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.6),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.6),
            nn.ReLU(),

            nn.Linear(512, 39)
            # nn.BatchNorm1d(256),
            # nn.Dropout(p=0.5),
            # nn.ReLU(),
            # nn.Linear(256, 39),            
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
if len(sys.argv) > 1:
    my_cuda = "cuda:{}".format(sys.argv[1])

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


# fix random seed for reproducibility
same_seeds(0)

# get device 
device = get_device()
print(f'DEVICE: {device}')

# the path where checkpoint saved
model_path = '/tmp2/b07902084/EEML/hw2/model-0.ckpt'

# # create model, define a loss function, and optimizer
# model = Classifier().to(device)
criterion = nn.CrossEntropyLoss() 
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)


VAL_RATIO = 0.2

percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))
train_size = train_x.shape[0]
val_size = val_x.shape[0]



# create testing dataset
test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
val_set = TIMITDataset(val_x, val_y)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

# create model and load weights from checkpoint
model = Classifier().to(device)
model.load_state_dict(torch.load(model_path))

predict = []
model.eval() # set the model to evaluation mode
with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader)):
        inputs = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability

        for y in test_pred.cpu().numpy():
            predict.append(y)

model.eval() # set the model to evaluation mode
val_acc = 0
val_loss = 0
with torch.no_grad():
    for i, data in enumerate(tqdm(val_loader)):
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

    print('Val Acc: {:3.6f} loss: {:3.6f}'.format(
         val_acc/len(val_set), val_loss/len(val_loader)
    ))

with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))
print("finish writing prediction")