import numpy as np
import re, math
import pandas as pd


def EUC(train_data, train_class, test_data, k):
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

def preprocess_prediction(df, column, index):
    average = []
    raw_data = np.array(df)
    for i in range(1, len(column)):
        if i != index:
            temp = raw_data[:, i]
            average.append(sum(temp)/len(temp))
    data = raw_data.tolist()
    for i in data:
        if i[0] == "M":
            i[0] = 1
        else:
            i[0] = 0
    classification = []
    for D in data:
        temp = D.pop(index)
        classification.append(temp)
    data = np.array(data)
    average = np.array(average)
    return data, classification, average

def prediction(input_dict, prediction):
    df = pd.read_csv('Data.csv')
    df = df.drop(columns=['id', 'Unnamed: 32'])
    column_list = list(df)
    temp_column_list = list(df)
    temp_column_list.pop(0)
    temp_column_list.pop(temp_column_list.index(prediction))
    test_data = []
    prediction_index = column_list.index(prediction)
    data, classification, average = preprocess_prediction(df, column_list, prediction_index)
    for i in range(len(column_list)):
        if column_list[i] == 'diagnosis':
            if input_dict['diagnosis'] == 'M':
                test_data.append(1)
            else:
                test_data.append(0)
        elif column_list[i] == prediction:
            pass
        elif column_list[i] in input_dict:
            test_data.append(input_dict[column_list[i]])
        else:
            test_data.append(average[temp_column_list.index(column_list[i])])
    result = EUC(data, classification, test_data, 7)
    return result

#input_dict = {'diagnosis': 'B', 'area_mean': 1001, 'area_se': 153.4, 'perimeter_mean': 122.8, 'area_worst': 2019}#'perimeter_worst': 184.6
#input_dict = {'diagnosis': 'M', 'area_mean': 566.3, 'area_se': 23.56, 'perimeter_worst': 99.7, 'perimeter_mean': 87.46}#'area_worst': 711.2,
#a = prediction(input_dict,'area_worst')
#print(a)
