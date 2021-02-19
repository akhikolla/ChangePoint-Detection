# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 01:07:39 2020

@author: 22602
"""

import glob

file = open('train_arg.txt', 'w')

train_path_list = list()
for dir_path in glob.glob("neuroblastoma-data/data/[sd]*/cv/*/testFolds/*"):
    train_path_list.append(dir_path)

for element in train_path_list:
     file.write(element)
     file.write('\n')
file.close()

file = open('plot_arg.txt', 'w')

plot_path_list = list()
for dir_path in glob.glob("neuroblastoma-data/data/[sd]*/cv/*"):
    plot_path_list.append(dir_path)

for element in plot_path_list:
     file.write(element)
     file.write('\n')
file.close()