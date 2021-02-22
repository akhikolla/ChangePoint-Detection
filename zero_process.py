
## import package
#from function import *
import sys
import os
import pandas as pd

## load the realating csv file
# get command line argument length.
#dir_path = "/Users/akhilachowdarykolla/Desktop/neuroblastoma-data/data/detailed"
dir_path ="neuroblastoma-data/data/detailed" #"/cv/sequenceID/testFolds/1"

## load the realating csv file
labels_path = dir_path + "/outputs.csv.xz"
input_path = dir_path + "/profiles.csv.xz"
output_path = dir_path

## load the file
dtypes = { "sequenceID": "category"}
profiles = pd.read_csv(input_path, dtype=dtypes)
labels = pd.read_csv(labels_path)

## extract all sequence id
sequenceID = labels["sequenceID"]
seq_data_list = []
# loop through all 
for id in sequenceID:
    #extract all data from profiels using same id
    one_object = profiles.loc[profiles["sequenceID"] == id]
    one_feature = one_object["signal"].tolist()
    seq_data_list.append(one_feature)

##padding data with zero
num_feature = max(map(len, seq_data_list))
print(num_feature)
sequenceID_pd = pd.DataFrame(sequenceID, columns = ['sequenceID'])
seq_data_pd = pd.DataFrame([xi+[0]*(num_feature-len(xi)) for xi in seq_data_list])
seq_data = pd.concat([sequenceID_pd, seq_data_pd], axis = 1)

#print the data as output
seq_data.to_csv(output_path + '/zero_inputs.csv.xz', index=False) 

