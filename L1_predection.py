# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 02:28:03 2020

@author: 22602
"""
import pandas as pd
import glob
from function import OutputFile
import sys

# get command line argument length.
argv = sys.argv[1]

## load the realating csv file
dir_path = argv + '/Inputs/'
L1_path = 'L1/'
folds_file = 'folds.csv'
L1_prediction_files = glob.glob(dir_path + L1_path + "/*.csv")

L1_pre_list = []
# for L1
for filename in L1_prediction_files:
    df = pd.read_csv(filename).iloc[:, -1].values
    L1_pre_list.append(df)

# this fucntion output the csv file
L1_output = pd.DataFrame(L1_pre_list[0])
L1_output = OutputFile(L1_output, L1_pre_list)
L1_output.to_csv(argv+ '/Outputs/L1Model.csv', index = None, header = False) 
