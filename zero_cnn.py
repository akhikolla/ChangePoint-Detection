# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 00:46:07 2020

@author: 22602
"""

## import package
#from function import *
import sys
import os
from sklearn import preprocessing
from early_stop import *
from zero_models import *
from impfun import *
import numpy as np

## load the realating csv file
# get command line argument length.
#dir_path = "/Users/akhilachowdarykolla/Desktop/neuroblastoma-data/data/detailed/cv/sequenceID/testFolds/1"
dir_path = "/Users/akhilachowdarykolla/Desktop/neuroblastoma-data/data/systematic/cv/sequenceID/testFolds/6"
model_id = "5"

## load the realating csv file
dir_path_split = dir_path.split("cv")
fold_path_split = dir_path.split("/testFolds/")
inputs_path = dir_path_split[0] + "zero_inputs.csv.xz"
labels_path = dir_path_split[0] + "outputs.csv.xz"
folds_path = fold_path_split[0] + "/folds.csv"
fold_num = int(fold_path_split[1])
outputs_path = dir_path + "/randomTrainOrderings/1/models/Cnn_zero_test/"# + model_id

inputs = pd.read_csv(inputs_path)
labels = pd.read_csv(labels_path)
folds = pd.read_csv(folds_path)

## processing data
labels = labels.values
num_feature = inputs.shape[1] - 1
seq_id = inputs.iloc[:, 0].to_frame()
Scale_inputs = preprocessing.scale(inputs.iloc[:, 1:])
Scale_inputs = pd.concat([seq_id, pd.DataFrame(Scale_inputs)], axis=1)
Scale_inputs = np.array(Scale_inputs)

folds = np.array(folds)
_, cor_index = np.where(Scale_inputs[:, 0, None] == folds[:, 0])
folds_sorted = folds[cor_index] # use for first split

#init the model parameter
criterion = SquareHingeLoss()
step = 2e-4
epoch = 1
model_id_int = int(model_id)
model = model_list[model_id_int]
optimizer = optim.Adam(model.parameters(),  lr= step)

#save the init model
model_path = "model_path/" + dir_path + "/" #+ model_id

#2 jobs give same reource?
if not os.path.exists(model_path):
    os.makedirs(model_path) 
PATH = model_path + '/cifar_net.pth'
torch.save(model.state_dict(), PATH)

# transfer data type
channel = 1
inputs, labels = Typetransfer_3D(Scale_inputs, labels, channel)

# split train test data
train_data, test_data, train_label, test_label = SplitFolder(inputs, labels, 
                                                    folds_sorted[:, 1], fold_num)

# get best epoch
best_epoch = earlyStop(model, optimizer, criterion, train_data, train_label, epoch).__call__()
print(best_epoch)
num_test = test_data.shape[0]

# init variables
model = model_list[model_id_int]
model.load_state_dict(torch.load(PATH))
optimizer = optim.Adam(model.parameters(),  lr= step)
    
_, test_outputs = Full(model, optimizer, criterion, 
                            train_data, train_label, test_data, 
                            test_label, best_epoch + 1).__call__()

# test data
with torch.no_grad():
    accuracy = 0
    for index in range(num_test):
        accuracy = accuracy + Accuracy(test_outputs[index], test_label[index].cpu().data.numpy())
    print(accuracy/num_test * 100)

# # this fucntion output the csv file
cnn_output = pd.DataFrame(test_outputs.cpu().data.numpy())
if not os.path.exists(outputs_path):
    os.makedirs(outputs_path) 
cnn_output.to_csv(outputs_path + '/predictions.csv') 




