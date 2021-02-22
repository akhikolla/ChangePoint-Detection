# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 09:34:25 2020

@author: 22602
"""
## import package
#from function import *
import sys
import os
from function import *
from sklearn import preprocessing
from spp_model import *

## load the realating csv file
# get command line argument length.
device = torch.device("cpu") 
split_list = []
#dir_path = "/Users/akhilachowdarykolla/Desktop/neuroblastoma-data/data/detailed/cv/sequenceID/testFolds/6" 
# for i in 1:
accurays = []
dir_path =sys.argv[1]
print(dir_path)

for j in range(0,10):
    print(model_list[j])

for i in range(0,10):
    #dir_path = sys.argv[1]
    print(dir_path) #"/Users/akhilachowdarykolla/Desktop/neuroblastoma-data/data/detailed/cv/sequenceID/testFolds/2"
    model_id = str(i)
    
    ## load the realating csv file
    dir_path_split = dir_path.split("cv")
    fold_path_split = dir_path.split("/testFolds/")
    profiles_path = dir_path_split[0] + "profiles.csv.xz"
    labels_path = dir_path_split[0] + "outputs.csv.xz"
    folds_path = fold_path_split[0] + "/folds.csv"
    fold_num = int(fold_path_split[1])
    outputs_path = dir_path + "/randomTrainOrderings/1/models/spp_test/" + model_id
    
    #init the model parameter
    criterion = SquareHingeLoss()
    step = 5e-5
    epoch = 10
    model_id_int = int(model_id)
    model = model_list[model_id_int]
    optimizer = optim.Adam(model.parameters(),  lr= step)
    min_feature = 500
    
    #save the init model
    model_path = "model_path/" + dir_path + "/" + model_id
    if not os.path.exists(model_path):
        os.makedirs(model_path) 
    PATH = model_path + 'cifar_net.pth'
    torch.save(model.state_dict(), PATH)
    
    ## load the file
    dtypes = { "sequenceID": "category"}
    profiles = pd.read_csv(profiles_path, dtype=dtypes)
    labels = pd.read_csv(labels_path)
    
    ## extract all sequence id
    sequenceID = labels["sequenceID"]
    seq_data_list = []
    # loop through all 
    for id in sequenceID:
        #extract all data from profiels using same id
        one_object = profiles.loc[profiles["sequenceID"] == id]
        one_feature = np.array(one_object["signal"].tolist())
        one_feature = preprocessing.scale(one_feature)
        #padding data less than 500
        N = one_feature.shape[0]
        if N < 500:
            padding_num = min_feature-N
            one_feature = np.pad(one_feature, (0, padding_num), 'constant')
            N = 500
        #transfter the data type
        one_feature = torch.from_numpy(one_feature.astype(float)).view(1, 1, N)
        one_feature = one_feature.type(torch.FloatTensor)
        one_feature = Variable(one_feature).to(device)
        #add to list
        seq_data_list.append(one_feature)
    inputs = seq_data_list
    
    ## get folder
    labels = labels.values
    folds = pd.read_csv(folds_path)
    folds = np.array(folds)
    _, cor_index = np.where(labels[:, 0, None] == folds[:, 0])
    folds_sorted = folds[cor_index] # use for first split
    
    ## transfer label type
    labels = torch.from_numpy(labels[:, 1:].astype(float))
    labels = labels.to(device).float()
    
    ## split train and test data
    bool_flag = folds_sorted[:, 1] == fold_num
    train_data = [a for i,a in enumerate(inputs) if not bool_flag[i]]
    test_data = [a for i,a in enumerate(inputs) if bool_flag[i]]
    train_label = labels[~bool_flag]
    test_label = labels[bool_flag]
    num_test = len(test_data)
    
    ## do early stop learning, get best epoch
    #split validation and subtraining data
    num_sed_fold = len(train_data)
    sed_fold = np.repeat([1,2,3,4,5], num_sed_fold/5)
    left = np.arange(num_sed_fold % 5) + 1
    sed_fold = np.concatenate((sed_fold, left), axis=0)
    np.random.shuffle(sed_fold)
    bool_flag = sed_fold == 1
    subtrain_data = [a for i,a in enumerate(train_data) if not bool_flag[i]]
    valid_data = [a for i,a in enumerate(train_data) if bool_flag[i]]
    subtrain_label = train_label[~bool_flag]
    valid_label = train_label[bool_flag]
    
    # do stochastic gradient descent
    valid_losses = []
    avg_valid_loss = []
    ## train the network
    for epoch in range(epoch):  # loop over the dataset multiple times
        for index, (data, label) in enumerate(zip(subtrain_data, subtrain_label)):
            model.train()  
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # do SGD
            outputs = model(data)
            loss = criterion(outputs, label)
            
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            for index, (data, label) in enumerate(zip(valid_data, valid_label)):
                model.eval()
                outputs = model(data)                    
                loss = criterion(outputs, label)
                valid_losses.append(loss.cpu().data.numpy())
        
        valid_loss = np.average(valid_losses)
        avg_valid_loss.append(valid_loss)
    
    #get best parameter
    min_loss_valid = min(avg_valid_loss)
    print(min_loss_valid)
    best_parameter = avg_valid_loss.index(min_loss_valid)
    
    print(best_parameter)
    
    # init variables for model
    model = model_list[model_id_int]
    model.load_state_dict(torch.load(PATH))
    optimizer = optim.Adam(model.parameters(),  lr= step)
    
    ## train the network using best epoch
    for epoch in range(best_parameter + 1): 
        for index, (data, label) in enumerate(zip(train_data, train_label)):
            model.train()  
        
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # do SGD
            outputs = model(data)
            loss = criterion(outputs, label)
            
            loss.backward()
            optimizer.step()
        
    test_losses = []
    test_outputs = []
    with torch.no_grad():
        for data in test_data:
            output = model(data).cpu().data.numpy().reshape(-1)
            test_outputs.append(output)
        
        for index, (data, label) in enumerate(zip(test_data, test_label)):
                model.eval()
                outputs = model(data)                    
                loss = criterion(outputs, label)
                test_losses.append(loss.cpu().data.numpy())
    
        print(np.average(test_losses))
            
        test_outputs = np.array(test_outputs)
            
    # test data
    with torch.no_grad():
        accuracy = 0
        for index in range(num_test):
            accuracy = accuracy + Accuracy(test_outputs[index], test_label[index].cpu().data.numpy())
            
    print(accuracy/num_test * 100)
    accurays.append(accuracy/num_test * 100)
    # this fucntion output the csv file
    cnn_output = pd.DataFrame(test_outputs)
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path) 
    cnn_output.to_csv(outputs_path + '/predictions.csv') 
