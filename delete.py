# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 16:15:01 2020

@author: 22602
"""
import sys
import shutil

## load the realating csv file
# get command line argument length.
dir_path = sys.argv[1]
fold_name = sys.argv[2]
outputs_path = dir_path + "/randomTrainOrderings/1/models/"

shutil.rmtree(outputs_path + fold_name)
print("finish")
