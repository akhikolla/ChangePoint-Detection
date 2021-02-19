# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 02:25:02 2020

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
baseline_path = 'baseline/'
folds_file = 'folds.csv'
baseline_files = glob.glob(dir_path + baseline_path + "/*.csv")

baseline_list = []
# for baseline
for filename in baseline_files:
    df = pd.read_csv(filename).iloc[:, -1].values
    baseline_list.append(df)

# this fucntion output the csv file
baseline_output = pd.DataFrame(baseline_list[0])
baseline_output = OutputFile(baseline_output, baseline_list)
baseline_output.to_csv(argv + '/Outputs/baselineModel.csv', index = None, header = False)

