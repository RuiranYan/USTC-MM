# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 10:02:33 2021

@author: lenovo
"""

import numpy as np
from scipy.sparse import coo_matrix
import random
import csv
import pandas as pd


# 抽样器
def random_index(rate):
    (start, index) = (0, 0)
    randnum = random.randint(1, sum(rate))
    for index, scope in enumerate(rate):
        start += scope
        if randnum <= start:
            break
    return index

def make_purchase(markov_matrix):
    Member_number = str(random.randint(1000, 5000))
    (day, month, year) = (random.randint(1, 29), random.randint(1, 12), 2020)
    (day, month, year) = (str(day), str(month), str(year))
    if int(day) < 10:
        day = '0' + day
    if int(month) < 10:
        month = '0' + month
    Date = day + '-' + month + '-' + year
    row_to_write=[]
    index = 0
    k=0
    while(True):
        k+=1
        index = random_index(markov_matrix[index])
        if index != 0:
            row_to_write.append([Member_number, Date,str(index+100)])
        else:
            break
    return row_to_write


if __name__ == '__main__':
    Data=pd.read_csv("matrix.csv")
    markov_matrix=Data.iloc[:,1:].values
    record_num = 20000
    with open("trial.csv", 'w') as csv_out_file:
        filewriter = csv.writer(csv_out_file, dialect='unix')
        header_to_write = ['Member_number', 'Date', 'itemDescription']
        filewriter.writerow(header_to_write)
        for i in range(0, record_num):
            row_to_write = make_purchase(markov_matrix)
            for j in range(len(row_to_write)):
                filewriter.writerow(row_to_write[j])