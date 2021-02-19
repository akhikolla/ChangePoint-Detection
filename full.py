# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 23:21:36 2020

@author: 22602
"""
from impfun import *
class Full:
    def __init__(self, model, optimizer, criterion, subtrain_data, subtrain_label, 
                 valid_data, valid_label, epoch):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.subtrain_data = subtrain_data
        self.subtrain_label = subtrain_label
        self.valid_data = valid_data
        self.valid_label = valid_label
        self.epoch = epoch
    
    def __call__(self):
        
        valid_losses = [];
        valid_outputs = [];
        ## train the network
        # loop through number of epoch
        for epoch in range(self.epoch):  
            self.model.train()   
    
            # zero the parameter gradients, init weight
            self.optimizer.zero_grad()
    
            # get output from our model, Y = f(wx)
            outputs = self.model(self.subtrain_data)
            # compute loss, loss = L(f(wx), y)
            loss = self.criterion(outputs, self.subtrain_label)
            # compute derivate, delta = derivate(Loss) with respect to w
            loss.backward()
            # update w. w = w – delta*w
            self.optimizer.step()
        
            with torch.no_grad():
               self.model.eval()
               valid_outputs = self.model(self.valid_data)                    
               valid_loss = self.criterion(valid_outputs, self.valid_label)

            valid_losses.append(valid_loss.cpu().data.numpy())
        
        return valid_losses, valid_outputs