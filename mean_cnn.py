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
from function import *

## load the realating csv file
# get command line argument length.
#dir_path = "/Users/akhilachowdarykolla/Desktop/neuroblastoma-data/data/detailed/cv/sequenceID/testFolds/6"
dir_path = sys.argv[1] 
#"/Users/akhilachowdarykolla/Desktop/neuroblastoma-data/data/systematic/cv/sequenceID/testFolds/6"

## load the realating csv file
dir_path_split = dir_path.split("cv")
fold_path_split = dir_path.split("/testFolds/")
inputs_path = dir_path_split[0] + "mean_inputs.csv.xz"
labels_path = dir_path_split[0] + "outputs.csv.xz"
folds_path = fold_path_split[0] + "/folds.csv"
fold_num = int(fold_path_split[1])
outputs_path = dir_path + "/randomTrainOrderings/1/models/Cnn_mean_pytorch/"

inputs = pd.read_csv(inputs_path)
labels = pd.read_csv(labels_path)
folds = pd.read_csv(folds_path)

## procssing data
labels = labels.values
num_feature = inputs.shape[1] - 1
seq_id = inputs.iloc[:, 0].to_frame()
Scale_inputs = preprocessing.scale(inputs.iloc[:, 1:])
Scale_inputs = pd.concat([seq_id, pd.DataFrame(Scale_inputs)], axis=1)
Scale_inputs = np.array(Scale_inputs)

folds = np.array(folds)
_, cor_index = np.where(Scale_inputs[:, 0, None] == folds[:, 0])
folds_sorted = folds[cor_index] # use for first split

# build the net work
class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 2, 9) 
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(2, 4, 9) 
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(228, 128)
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = F.max_pool2d(x, (1, 4))
        x = self.layer2(x)
        x = F.max_pool2d(x, (1, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#init the model parameter
cnn_model = convNet().to(device)
criterion = SquareHingeLoss()
step = 1e-4
epoch = 1
model = convNet().to(device)
optimizer = optim.Adam(model.parameters(),  lr= step)

#save the init model
if not os.path.exists("model_path/" + dir_path):
    os.makedirs("model_path/" + dir_path) 
PATH = "model_path/" + dir_path + 'cifar_net.pth'
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
model = convNet().to(device)
model.load_state_dict(torch.load(PATH))
optimizer = optim.Adam(model.parameters(),  lr= step)
    
_, test_outputs = Full(model, optimizer, criterion, 
                            train_data, train_label, test_data, 
                            test_label, best_epoch+1).__call__()


# test data
with torch.no_grad():
    accuracy = 0
    for index in range(num_test):
        accuracy = accuracy + Accuracy(test_outputs[index], test_label[index].cpu().data.numpy())
    print(accuracy/num_test * 100)

# # this fucntion output the csv file
cnn_output = pd.DataFrame(test_outputs.cpu().data.numpy())
if not os.path.exists(outputs_path + "5"):
     os.makedirs(outputs_path) 
cnn_output.to_csv(outputs_path + 'predictions.csv')









