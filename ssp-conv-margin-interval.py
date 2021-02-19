#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch
import math
from torch import optim
from torch.autograd import Variable


# In[2]:


root_path = './source-data/'
sub_path = 'detailed/'   # change this when it is necessary
spilits_path = 'cv/profileID/'


# In[3]:


data = pd.read_csv(root_path+sub_path+'profiles.csv.xz')
data_target = pd.read_csv(root_path+sub_path+'outputs.csv.xz').iloc[:,-2:].values
data_splits = pd.read_csv(root_path+sub_path+spilits_path+'folds.csv').iloc[:,-1].values


# In[4]:


name_list = data['sequenceID'].unique()
name_list.sort()
data_dym_storage = []
for name in name_list:
    data_dym_storage.append(data[data['sequenceID'] == name]['signal'].values)
data_dym_storage = np.array(data_dym_storage)


# In[5]:


def data_splits_do(fold_id):
    tfmarker = (data_splits == fold_id)
    data_train = data_dym_storage[~tfmarker]
    data_trtar = data_target[~tfmarker]
    data_test = data_dym_storage[tfmarker]
    data_tstar = data_target[tfmarker]
    return data_train, data_trtar, data_test, data_tstar


# In[6]:


data_train, data_trtar, data_test, data_tstar = data_splits_do(1)


# In[7]:


class SquareHingeLoss(nn.Module):
    def __init__(self):
        super(SquareHingeLoss,self).__init__()
    
    def ifelse(self, condition, a, b):
        crit = (condition >= 0).squeeze(1)
        copy_con = condition.clone()
        copy_con[crit] = condition[crit] ** 2
        copy_con[~crit] = b
        return copy_con

    def phi(self, in_phi):
        return self.ifelse(in_phi, in_phi**2, 0) 
       
    def forward(self, x, target_y):
#         print(torch.mean(self.phi(-x + target_y[:,:,0] + 1) + self.phi(x - target_y[:,:,1] + 1)))
        return torch.mean(self.phi(- x + target_y[:,:,0] + 1) + self.phi(x - target_y[:,:,1] + 1))


# In[8]:


# This one based on the AdaptiveMax/Avg method built in torch

class SpatialPyramidPooling(nn.Module):
    """Generate fixed length representation regardless of image dimensions
    Based on the paper "Spatial Pyramid Pooling in Deep Convolutional Networks
    for Visual Recognition" (https://arxiv.org/pdf/1406.4729.pdf)
    :param [int] num_pools: Number of pools to split each input feature map into.
        Each element must be a perfect square in order to equally divide the
        pools across the feature map. Default corresponds to the original
        paper's implementation
    :param str mode: Specifies the type of pooling, either max or avg
    """

    def __init__(self, num_pools=[1, 4, 16], mode='max'):
        super(SpatialPyramidPooling, self).__init__()
        self.name = 'SpatialPyramidPooling'
        if mode == 'max':
            pool_func = nn.AdaptiveMaxPool1d
        elif mode == 'avg':
            pool_func = nn.AdaptiveAvgPool1d
        else:
            raise NotImplementedError(f"Unknown pooling mode '{mode}', expected 'max' or 'avg'")
        self.pools = []
        for p in num_pools:
            self.pools.append(pool_func(p))

    def forward(self, feature_maps):
        """Pool feature maps at different bin levels and concatenate
        :param torch.tensor feature_maps: Arbitrarily shaped spatial and
            channel dimensions extracted from any generic convolutional
            architecture. Shape ``(N, C, W)``
        :return torch.tensor pooled: Concatenation of all pools with shape
            ``(N, C, sum(num_pools))``
        """
        assert feature_maps.dim() == 3, 'Expected 3D input of (N, C, W)'
        batch_size = feature_maps.size(0)
        channels = feature_maps.size(1)
        pooled = []
        for p in self.pools:
            pooled.append(p(feature_maps).view(batch_size, channels, -1))
        return torch.cat(pooled, dim=2)


# In[10]:


class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        self.spp = SpatialPyramidPooling()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 8, 5, 3), # 8 x 332
            nn.ReLU(True),
            nn.Conv1d(8, 16, 3, 2, 1), # 16 x 166
            nn.ReLU(True),
            nn.Conv1d(16, 32, 3, 2, 1), # 32 x 83
            nn.ReLU(True),
            nn.Conv1d(32, 32, 3, 2, 1), # 32 x 42
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(32*21, 128),
            nn.ReLU(True),
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = x.float()
        x = self.layer1(x)
        x = self.spp(x)
        x = x.reshape(x.size(0),-1)
        x = self.layer2(x)
        return x


# In[14]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = convNet().to(device)
criterion = SquareHingeLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# In[21]:


torch.from_numpy(data_train[0]).view(1,1,-1)


# In[ ]:


e = 0
num_epoches = 3000
train_loss_record = np.zeros(num_epoches)
train_acc_record = np.zeros(num_epoches)
test_loss_record = np.zeros(num_epoches)
test_acc_record = np.zeros(num_epoches)
for epoch in range(num_epoches):
    loss_value, iter_num, print_loss = 0, 0, 0
    acc = 0
    for i, (data, target) in enumerate(zip(data_train, data_trtar)):
        e += 1
        iter_num += 1
        data = torch.from_numpy(data).view(1, 1, -1)
        data = Variable(data).to(device)
        target = torch.from_numpy(target).view(-1, 1)
        target = Variable(target).to(device)
        inputs = inputs.type(torch.DoubleTensor).to(device)
        out = model(inputs)
        loss = criterion(out, targets.float())
        optimier.zero_grad()
        loss.backward()
        optimier.step()
        
        print_loss += loss.cpu().data.numpy()
        acc += accuarcy(out.cpu().data, targets.cpu().data.float()).data.numpy()
    
    test_in = Variable(valdataset.tensors[0]).to(device)
    test_in = test_in.type(torch.DoubleTensor).to(device)
    test_out = model(test_in)
    test_loss = criterion(test_out, Variable(valdataset.tensors[1].cuda()).float())
    test_loss = test_loss.cpu().data.numpy()
    test_acc = accuarcy(test_out.cpu().data, valdataset.tensors[1].float()).data.numpy()
        
    print('-'* 120)
    print('Epoch [{:-03d}/{}]  |  Train Loss:  {:.3f}  |  Test Loss:  {:.3f}  |  Test Accuarcy:  {:.3f}  |  Train Accuracy:  {:.3f}'
          .format(epoch+1, num_epoches, print_loss/iter_num, test_loss, test_acc, acc/iter_num))
    train_loss_record[epoch] = print_loss/iter_num
    test_loss_record[epoch] = test_loss
    train_acc_record[epoch] = acc/iter_num
    test_acc_record[epoch] = test_acc

