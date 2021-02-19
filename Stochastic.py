# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 23:53:25 2020

@author: 22602
"""

from function import *
class Stochastic:
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
         
        valid_losses = []
        avg_valid_loss = []

        ## train the network
        for epoch in range(self.epoch):  # loop over the dataset multiple times
            for index, (data, label) in enumerate(zip(self.subtrain_data, self.subtrain_label)):
                self.model.train()  
        
                # zero the parameter gradients
                self.optimizer.zero_grad()
        
                # do SGD
                outputs = self.model(data)
                loss = self.criterion(outputs, label)
                
                loss.backward()
                self.optimizer.step()
            
        
            with torch.no_grad():
               for index, (data, label) in enumerate(zip(self.valid_data, self.valid_label)):
                    self.model.eval()
                    outputs = self.model(data)                    
                    loss = self.criterion(outputs, label)
                    valid_losses.append(loss.cpu().data.numpy())
            
            valid_loss = np.average(valid_losses)
            avg_valid_loss.append(valid_loss)
        
        # get valid data
        with torch.no_grad():
            self.model.eval()  
            valid_outputs = self.model(self.valid_data).cpu().data.numpy()         
        
        return avg_valid_loss, valid_outputs