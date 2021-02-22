# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 04:12:55 2020

@author: 22602
"""

## import package
#from function import *
import sys
import os
from impfun import *

## load the realating csv file
# get command line argument length.

#dir_path = "/Users/akhilachowdarykolla/Desktop/neuroblastoma-data/data/detailed/"
dir_path = "neuroblastoma-data/data/detailed"

## load the realating csv file
dir_path_split = dir_path.split("/cv/")
labels_path = dir_path_split[0] + "/outputs.csv.xz"
input_path = dir_path_split[0] + "/profiles.csv.xz"
output_path = dir_path_split[0]

#init the mean pool
fix_feature = 500
pooler = nn.AdaptiveAvgPool1d(fix_feature)

## load the file
dtypes = { "sequenceID": "category"}
profiles = pd.read_csv(input_path, dtype=dtypes)
labels = pd.read_csv(labels_path)

## extract all sequence id
sequenceID = labels["sequenceID"]
seq_data_list = []
# loop through all 
print(len(sequenceID))
num_itr = len(sequenceID)
for id in sequenceID:
    #extract all data from profiels using same id
    one_object = profiles.loc[profiles["sequenceID"] == id]
    one_feature = torch.from_numpy(np.array(one_object["signal"]))
    ##mean the data feature
    num_feature = one_feature.size()[0]
    one_feature = one_feature.reshape(1,1,num_feature)
    mean_feature = pooler(one_feature).data.numpy()
    mean_feature = pd.DataFrame(mean_feature.reshape(1,fix_feature))
    seq_data_list.append(mean_feature)


##convert data yupt
seq_data_list = np.asarray(seq_data_list)
seq_data_list=seq_data_list.reshape(num_itr,500)
#print(np.ravel(seq_data_list)) 
seq_data_pd = pd.DataFrame(seq_data_list)
sequenceID_pd = pd.DataFrame(sequenceID, columns = ['sequenceID'])
seq_data = pd.concat([sequenceID_pd, seq_data_pd], axis = 1)

#print the data as output
seq_data.to_csv(output_path + '/mean_inputs.csv.xz', index=False) 
