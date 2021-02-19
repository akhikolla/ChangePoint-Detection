# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 01:34:41 2020

@author: 22602
"""
from impfun import *


class convNet_2(nn.Module):
    def __init__(self):
        super(convNet_2, self).__init__()
        self.pool1 =  nn.MaxPool1d(4)
        self.pool2 =  nn.MaxPool1d(3)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 4, 9) 
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 6, 9)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(2946, 500)
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(500, 100)
        )
        
        self.layer5 = nn.Sequential(
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class convNet_2_act(nn.Module):
    def __init__(self):
        super(convNet_2_act, self).__init__()
        self.pool1 =  nn.MaxPool1d(4)
        self.pool2 =  nn.MaxPool1d(3)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 4, 9),
            nn.RReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 6, 9),
            nn.RReLU()
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(2946, 500)
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(500, 100)
        )
        
        self.layer5 = nn.Sequential(
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
class convNet_3(nn.Module):
    def __init__(self):
        super(convNet_3, self).__init__()
        self.pool1 =  nn.MaxPool1d(4)
        self.pool2 =  nn.MaxPool1d(3)
        self.pool3 =  nn.MaxPool1d(2)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 4, 9) 
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 6, 9)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(6, 8, 9)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(1928, 500)
        )
        
        self.layer5 = nn.Sequential(
            nn.Linear(500, 100)
        )
        
        self.layer6 = nn.Sequential(
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

device = torch.device("cpu") 
class convNet_3_act(nn.Module):
    def __init__(self):
        super(convNet_3_act, self).__init__()
        self.pool1 =  nn.MaxPool1d(4)
        self.pool2 =  nn.MaxPool1d(3)
        self.pool3 =  nn.MaxPool1d(2)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 4, 9),
            nn.RReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 6, 9),
            nn.RReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(6, 8, 9),
            nn.RReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(1928, 500)
        )
        
        self.layer5 = nn.Sequential(
            nn.Linear(500, 100)
        )
        
        self.layer6 = nn.Sequential(
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class convNet_4(nn.Module):
    def __init__(self):
        super(convNet_4, self).__init__()
        self.pool1 =  nn.MaxPool1d(4)
        self.pool2 =  nn.MaxPool1d(3)
        self.pool3 =  nn.MaxPool1d(2)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 4, 9) 
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 6, 9)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(6, 8, 9)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(8, 10, 9)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(1580, 500)
        )
        
        self.layer6 = nn.Sequential(
            nn.Linear(500, 100)
        )
        
        self.layer7 = nn.Sequential(
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool2(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        x = self.layer4(x)
        x = self.pool3(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class convNet_4_act(nn.Module):
    def __init__(self):
        super(convNet_4_act, self).__init__()
        self.pool1 =  nn.MaxPool1d(4)
        self.pool2 =  nn.MaxPool1d(3)
        self.pool3 =  nn.MaxPool1d(2)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 4, 9),
            nn.RReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 6, 9),
            nn.RReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(6, 8, 9),
            nn.RReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(8, 10, 9),
            nn.RReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Linear(1580, 500)
        )
        
        self.layer6 = nn.Sequential(
            nn.Linear(500, 100)
        )
        
        self.layer7 = nn.Sequential(
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool2(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        x = self.layer4(x)
        x = self.pool3(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

model_list = []
model_class = [convNet_2(), convNet_2_act(), convNet_3(), 
               convNet_3_act(),convNet_4(), convNet_4_act()]
for model in model_class:
    model_list.append(model.to(device))