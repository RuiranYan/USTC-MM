import numpy as np
from scipy.sparse import coo_matrix
import random
import csv
import pandas as pd

def get_markov_matrix(nrow, ncol, density):
    num = int(nrow * ncol * density)
    row = np.zeros(num)
    col = np.zeros(num)
    data = np.zeros(num)
    for i in range(num):
        row[i] = random.randint(0, nrow - 1)
        col[i] = random.randint(0, ncol - 1)
        if row[i]<col[i]:
            data[i] = random.randint(1, 100)
        else:
            data[i] = 0
    coo = coo_matrix((data, (row, col)), shape=(nrow, ncol))
    sparse = coo.toarray()

    markov_mat = np.zeros((nrow + 1, ncol + 1))
    count = 0
    for item in sparse:
        count += 1
        newrow = item
        if np.sum(newrow) > 0:
            stop = np.array([2*np.sum(newrow)]).astype(int)
        else:
            stop = np.array([100])
        new = np.concatenate((stop, newrow*10))
        markov_mat[count] = new
    # 假设最开始买的物品频率相等
    lambd=0.08
    x = np.arange(0,nrow,1)
    y = lambd*np.exp(-lambd*x)
    for i in range(ncol+1):
        markov_mat[i, (i+1):] += (y[i:]*100000).astype(int)
        if i!=0:
            markov_mat[i,0]+=sum((y[i:]*50000).astype(int))
    print(markov_mat)
    return markov_mat

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
    while(True):
        index = random_index(markov_matrix[index])
        if index != 0:
            row_to_write.append([Member_number, Date,str(index+100)])
        else:
            break
    return row_to_write

if __name__ == '__main__':
    (num_goods, density) = (100, 0.02)
    goods = [i for i in range(100, num_goods + 100)]
    markov_matrix = get_markov_matrix(num_goods, num_goods, density)
    Data=pd.DataFrame(markov_matrix)
    print(Data)
    Data.to_csv("matrix.csv")
    #record_num = 20000
    #with open("trial.csv", 'w') as csv_out_file:
       # filewriter = csv.writer(csv_out_file, dialect='unix')
       # header_to_write = ['Member_number', 'Date', 'itemDescription']
       # filewriter.writerow(header_to_write)
       # for i in range(0, record_num):
        #    row_to_write = make_purchase(markov_matrix)
        #    for j in range(len(row_to_write)):
           #     filewriter.writerow(row_to_write[j])