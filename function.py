## import package
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device("cpu") 

# this fucntion we transfer the data and label type from numpy to tensor
def Typetransfer_2D(data, label):
    data = torch.from_numpy(data[:, 1:].astype(float))
    data = data.type(torch.FloatTensor)
    data = Variable(data).to(device)
    label = torch.from_numpy(label[:, 1:].astype(float))
    label = label.to(device).float()
    
    return data, label

# this fucntion we transfer the data and label type from numpy to tensor
def Typetransfer_3D(data, label, channel):
    num_data = data.shape[0]
    num_feature = data.shape[1] - 1
    data = torch.from_numpy(data[:, 1:].astype(float)).view(num_data, channel, num_feature)
    data = data.type(torch.FloatTensor)
    data = Variable(data).to(device)
    label = torch.from_numpy(label[:, 1:].astype(float))
    label = label.to(device).float()
    
    return data, label

## split data by folder
# function to split data
def SplitFolder(inputs, labels, folders, fold_id):
    bool_suq = folders == fold_id
    train_data = inputs[~bool_suq]
    test_data = inputs[bool_suq]
    train_label = labels[~bool_suq]
    test_label = labels[bool_suq]

    return train_data, test_data, train_label, test_label

## accuracy function
def Accuracy(predicated_y, target_y):
    if (np.logical_and(target_y[0] - predicated_y <= 0,
                         predicated_y - target_y[1] <= 0)):
        return 1
    else:
        return 0
    
## squareHangLoss function
class SquareHingeLoss(nn.Module):
    def __init__(self):
        super(SquareHingeLoss,self).__init__()
       
    def forward(self, predicated_y, target_y):
        num = predicated_y.size()[0]
        
        if (num == 1):
            area = torch.cat((predicated_y - target_y[0], target_y[1] - predicated_y) , 0)
        
        else:
            predicated_y = predicated_y.view(-1)
            area = torch.cat((predicated_y - target_y[: ,0], target_y[: ,1] - predicated_y) , 0)
    
        crit = (area - 1 < 0)
        area[crit] = (area[crit] - 1) ** 2
        area[~crit] = torch.tensor([0.0], device='cpu', requires_grad=True)
        result = torch.mean(area)
        
        return result
    
class MyHingeLoss(nn.Module):
    def __init__(self):
        super(MyHingeLoss,self).__init__()
       
    def forward(self, predicated_y, target_y):
        num = predicated_y.size()[0]
        
        if (num == 1):
            area = torch.cat((predicated_y - target_y[0], target_y[1] - predicated_y) , 0)
        
        else:
            predicated_y = predicated_y.view(-1)
            area = torch.cat((predicated_y - target_y[: ,0], target_y[: ,1] - predicated_y) , 0)
    
        crit = (area < 0)
        area[crit] = (area[crit]) ** 2
        area[~crit] = torch.tensor([0.0], device='cpu', requires_grad=True)
        result = torch.mean(area)
        
        return result

# ssp function,based on the AdaptiveMax/Avg method built in torch
class SpatialPyramidPooling(nn.Module):
    def __init__(self, mode):
        super(SpatialPyramidPooling, self).__init__()
        num_pools = [1, 4, 16, 36]
        self.name = 'SpatialPyramidPooling'
        self.mode = mode
        if mode == 'max' or mode == 'min':
            pool_func = nn.AdaptiveMaxPool1d
        elif mode == 'avg':
            pool_func = nn.AdaptiveAvgPool1d
        else:
            raise NotImplementedError(f"Unknown pooling mode '{mode}', expected 'max' or 'avg'")
        self.pools_fun = []
        for p in num_pools:
            self.pools_fun.append(pool_func(p))

    def forward(self, feature_maps):
        pooled = []
        for pool_fun in self.pools_fun:
            if self.mode == 'min':
                pooled.append(pool_fun(-feature_maps) * (-1))
            else:
                pooled.append(pool_fun(feature_maps))
        return torch.cat(pooled, dim=2)


# this is l1 regularizer
class L1Regularizer:
    """
    L1 regularized loss
    """
    def __init__(self, model, loss_val, lambda_rate):
        self.model = model
        self.loss_val = loss_val
        self.lambda_rate = lambda_rate
    
    def add_l1(var):
        return var.abs().sum()

    def regularized_param(self):
        for model_param_name, model_param_value in self.model.named_parameters():
            if model_param_name.endswith('weight'):
                self.loss_val += self.lambda_rate * L1Regularizer.add_l1(model_param_value)
                
        return self.loss_val
    

