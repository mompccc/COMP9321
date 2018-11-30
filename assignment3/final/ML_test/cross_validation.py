import numpy as np
import matplotlib.pyplot as plt
import re, math
import pandas as pd


def EUC(train_data, train_class, test_data, k):
    # this is weighted prediction
    instances = train_data.shape[0]
    minus = np.tile(test_data, (instances, 1)) - train_data  # step 1: minus
    squared_m = minus ** 2  # step 2: square
    squared_dist = squared_m.sum(axis=1)  # step 3: sum up
    distance = squared_dist ** 0.5  # step 4: extraction of a root
    sorted_index = np.argsort(distance)  # step 5: sort the distance in ascending and return rank list
    all_price = []  # list to store all nearest k prices
    denominator = 0
    mu = sum(distance)/len(distance)
    for i in range(k):
        ratio = (1 / mu * (math.pi * 2) ** 0.5) * math.e ** (-(distance[sorted_index[i]]) ** 2 / (2 * mu ** 2)) # compute the ratio of current point distance and basement point distance
        temp_price = train_class[sorted_index[i]] * ratio  # plus weighted price
        all_price.append(temp_price)
        denominator += ratio
    return sum(all_price)/denominator


def cross_validation(file, K, L, C):
    result = []
    for n in range(L):  # repeat L times and get each line tested
        train_data = []
        train_class = []
        temp_list = list(file[n])
        test_data = temp_list[:-1]
        for i in range(L):
            if i != n:  # set all lines except the test line into training data
                temp_list = list(file[i])
                train_class.append(temp_list.pop(24))
                train_data.append(temp_list)
        train_data = np.array(train_data)
        result.append(EUC(train_data, train_class, test_data, K))  # change EUC to EUC1 to implement weighted kNN
    out = []  # a list to store all error rate
    temp_minus = []  # a list to store D-value for later root-mean-square error computation
    for i in range(L):
        if result[i] < C[i]:
            temp_minus.append(C[i]-result[i])  # root-mean-square error computation step 1: D-value
            per = (C[i]-result[i])/C[i]  # error rate computation step 1
        else:
            temp_minus.append(result[i] - C[i])
            per = (result[i] - C[i])/C[i]
        out.append(per)
    sum_per = sum(out)/len(out) * 100  # get average error rate

    A = 0
    for r in temp_minus:
        A += r**2  # root-mean-square error computation step 2: square
    V = (A/len(temp_minus))**0.5  # root-mean-square error computation step 3: extraction of a root
    return sum_per, V  # average error rate, root-mean-square error

def read_csv(file_name):
    f = open(file_name, 'r')
    content = f.read()
    final_list = list()
    rows = content.split('\n')
    for row in rows:
        if row != "":
            final_list.append(row.split(','))
    a=final_list.pop(0)
    print('feature',a[25])
    for i in range(0,len(final_list)):
        final_list[i].pop(0)
        for j in range(0,len(final_list[i])):
            if final_list[i][j] == "M":
                final_list[i][j] = 1
            elif final_list[i][j] == "B":
                final_list[i][j] = 0
            else:
                final_list[i][j] = float(final_list[i][j])
    return final_list



data_file = read_csv('data.csv')
length = len(data_file)
classification = []

count = 0
for x in data_file:
    temp = list(x)
    classification.append(temp.pop(24))
x_aix = []
y_aix0 = []
y_aix1 = []
for k in range(1, 30):
    x_aix.append(k)
    result = cross_validation(data_file, k, length, classification)
    y_aix0.append(result[0])
    y_aix1.append(result[1])
    print("cross-validation: k={}, error={}, RMSE={}".format(k, result[0], result[1]))
fig,picture1 = plt.subplots()
picture2 = picture1.twinx()
picture1.plot(x_aix, y_aix0, 'bp--')
picture1.set_ylabel('Average  Deviation  Error  Rate %')
picture1.set_xlabel('K  value')
picture1.set_title('Cross-Validation of area_worst')
picture2.plot(x_aix, y_aix1, 'rp:')
picture2.set_ylabel("RMSE  for  each  K")
plt.show()
