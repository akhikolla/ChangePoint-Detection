import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
import os

## accuracy function
def Accuracy(predicated_y, target_y):
    if (np.logical_and(target_y[0] - predicated_y <= 0,
                         predicated_y - target_y[1] <= 0)):
        return 1
    else:
        return 0


# function to split data
def SplitFolder(labels, folders, fold_id):
    bool_suq = folders == fold_id
    train_label = labels[~bool_suq][:, 1:]
    test_label = labels[bool_suq][:, 1:]

    return train_label, test_label

# get command line argument length.

dir_path = sys.argv[1]

#"/Users/akhilachowdarykolla/Desktop/neuroblastoma-data/data/systematic/cv/sequenceID"

## load the realating csv file
dir_path_split = dir_path.split("/cv/")
labels_path = dir_path_split[0] + "/outputs.csv.xz"
folds_path = dir_path + "/folds.csv"
input_path = dir_path + "/testFolds"
output_path = dir_path
main_path_split = dir_path_split[0].split("/data/")

labels = pd.read_csv(labels_path)
folds = pd.read_csv(folds_path)

labels = labels.values
folds = np.array(folds)
_, cor_index = np.where(labels[:, 0, None] == folds[:, 0])
folds_sorted = folds[cor_index] # use for first split

model_name_list = []
# get name of all the model
for py in glob.glob( input_path + "/1/randomTrainOrderings/1/models/*"):
    #get model name
    file_name = os.path.basename(py)
    name, _ = os.path.splitext(file_name)
    model_name_list.append(name)

num_model = len(model_name_list)
print("Model list")
print(model_name_list)

#loop through model name list, for each list, create the relating file
model_list = []
for name in model_name_list:
    file_list = []
    if(name == "spp")
    path = glob.glob( input_path + "/*/randomTrainOrderings/1/models/" 
                          + name + "/predictions.csv")
    for file_path in path:
        #print(file_path)
        #get last column of each file
        df = pd.read_csv(file_path).iloc[:, -1].values
        file_list.append(df)
    num_test = len(file_list)
    #print(num_test)
    #print(file_list)   
    accuracy_list = []
    # calculate accuracy
    for fold_num in range(num_test):
        _, fold_lab = SplitFolder(labels, folds_sorted[:, 1], fold_num + 1)
        acc = 0
        for (data, label) in zip(file_list[fold_num], fold_lab):
            #print(data)
            label = label.reshape(2)
            acc += Accuracy(data, label)
        
        num = fold_lab.shape[0]
        accuracy_list.append(acc/num * 100)
    
    average = sum(accuracy_list)/len(accuracy_list)
    model_component = [accuracy_list, name, average]
    model_list.append(model_component)
model_list.sort(key = lambda model_list: model_list[2]) 
model_list = np.array(model_list,dtype=object)
model_accuracy = model_list[:, 0]
model_name = model_list[:, 1]

#print(num_model)
print(model_accuracy[index])
#print(num_test * [model_name[index]])
# for index in range(num_model):
#     plt.scatter(model_accuracy[index], num_test * [model_name[index]], color = "black")
# plt.xlabel("accuracy.percent %")
# plt.ylabel("algorithm")
# plt.tight_layout()
# main_name = main_path_split[1]
# sub_name = dir_path_split[1]
# plt.savefig('/Users/akhilachowdarykolla/Desktop/plot_folder/' + main_name + '_' + sub_name + '.png')
# #plt.savefig("SS_linear_accuracy.png")
# plt.title(main_name + '_' + sub_name)

for index in range(num_model):
    plt.scatter(model_accuracy[index], num_test * [model_name[index]], color = "black")
plt.xlabel("accuracy.percent %")
plt.ylabel("algorithm")
plt.tight_layout()
main_name = main_path_split[1]
sub_name = dir_path_split[1]
plt.savefig('plot_folder/' + main_name + '_' + sub_name + '.png')
#plt.savefig("SS_linear_accuracy.png")
plt.title(main_name + '_' + sub_name)
    
