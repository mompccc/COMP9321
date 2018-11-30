from flask import Flask, render_template, session, request, redirect, url_for
import os
import numpy as np
import re, math
import pandas as pd
from sklearn import svm


app = Flask(__name__)

pre_list = ['diagnosis','area_mean','area_se','area_worst','perimeter_mean','perimeter_worst']
pre_dic = {'diagnosis':'pre_dia','area_mean':"pre_am",'area_se':"pre_as",'area_worst':"pre_aw",'perimeter_mean':"pre_pm",'perimeter_worst':"pre_pw"}
class_dic = {'area_mean':"cla_am",'area_se':"cla_as",'area_worst':"cla_aw",'perimeter_mean':"cla_pm",'perimeter_worst':"cla_pw"}
class_list = ['area_mean','area_se','area_worst','perimeter_mean','perimeter_worst']
description_dict = {'idID': 'number', 'Diagnosis': 'The diagnosis of breast tissues (M = malignant, B = benign)', 'Radius Mean': 'mean of distances from center to points on the perimeter',
                    'Texture Mean': 'standard deviation of gray-scale values', 'Perimeter Mean': 'mean size of the core tumor', 'Area Mean': 'mean size of the measured area','Smoothness Mean': 'mean of local variation in radius lengths',
                    'Compactness Mean': 'mean of perimeter^2 / area - 1.0', 'Concavity Mean': 'mean of severity of concave portions of the contour','Concave Points Mean': 'mean for number of concave portions of the contour',
                    'Symmetry Mean': '','Fractal Dimension Mean':'mean for "coastline approximation" - 1','Radius Se':'standard error for the mean of distances from center to points on the perimeter',
                    'Texture SE':'standard error for standard deviation of gray-scale values', 'Perimeter SE':'','Area SE':'standard error of the measured area size', 'Smoothness SE':'standard error for local variation in radius lengths',
                    'Compactness SE':'standard error for perimeter^2 / area - 1.0','Concavity SE':'standard error for severity of concave portions of the contour','Concave Points SE':'standard error for number of concave portions of the contour',
                    'Symmetry SE':'','Fractal Dimension SE':'standard error for "coastline approximation" - 1','Radius Worst':'"worst" or largest mean value for mean of distances from center to points on the perimeter',
                    'Texture Worst':'"worst" or largest mean value for standard deviation of gray-scale values','Perimeter Worst':'worst size of the core tumor','Area Worst':'worst value of measured area size','Smoothness Worst':'"worst" or largest mean value for local variation in radius lengths',
                    'Compactness Worst': '"worst" or largest mean value for perimeter^2 / area - 1.0','Concavity Worst':'"worst" or largest mean value for severity of concave portions of the contour',
                    'Concave Points Worst':'"worst" or largest mean value for number of concave portions of the contour','Symmetry Worst':'','Fractal Dimension Worst':'"worst" or largest mean value for "coastline approximation" - 1'}

@app.route('/', methods=['GET','POST'])
def ass3():
    error=None
    return render_template('index.html', status=0, which='class', description_dict=description_dict, error=error)


@app.route('/calculate', methods=['GET', 'POST'])
def calculate():
    para = request.form
    final_result = []
    final_value_dict = {}
    final_value_dict_new = {}
    calculate_dict = {}
    which_one = ''
    error=None
    if para['if_pre'] == "1":
        if para['pre_word'] == 'none':
            error='error1'
            return render_template('index.html', status=0, which='pre', description_dict=description_dict,error=error)
        elif 'pre_dia' not in para:
            error = 'error2'
            return render_template('index.html', status=0, which='pre', description_dict=description_dict, error=error)
        for i in pre_list:
            if pre_dic[i] in para:
                final_value_dict[i] = para[pre_dic[i]]
                if re.findall(r'[0-9]+', para[pre_dic[i]]):
                    calculate_dict[i] = float(para[pre_dic[i]])
                else:
                    calculate_dict[i] = para[pre_dic[i]]
        pre_word = para['pre_word']
        result = prediction(calculate_dict, pre_word)
        final_result.append(pre_word)
        final_result.append(round(result, 2))
        which_one = "pre"
    elif para['if_clas'] == "1":
        for i in class_list:
            if para[class_dic[i]] != '':
                calculate_dict[i] = float(para[class_dic[i]])
                final_value_dict[i] = para[class_dic[i]]
        result = classify(calculate_dict)[0]
        final_result.append('diagnosis')
        if result == "M":
            final_result.append('Malignant')
        else:
            final_result.append('Benign')
        which_one = "class"
    if which_one == "pre":
        final_value_dict_new['Diagnosis'] = final_value_dict['diagnosis']
    for key in final_value_dict.keys():
        if re.findall(r'area_mean', key):
            final_value_dict_new['Area Mean'] = final_value_dict['area_mean']
        elif re.findall(r'area_se', key):
            final_value_dict_new['Area SE'] = final_value_dict['area_se']
        elif re.findall(r'area_worst', key):
            final_value_dict_new['Area Worst'] = final_value_dict['area_worst']
        elif re.findall(r'perimeter_mean', key):
            final_value_dict_new['Perimeter Mean'] = final_value_dict['perimeter_mean']
        elif re.findall(r'perimeter_worst', key):
            final_value_dict_new['Perimeter Worst'] = final_value_dict['perimeter_worst']
    return render_template('index.html', status=1, final_result=final_result, final_value_dict=final_value_dict_new, which=which_one, description_dict=description_dict, error=error)


#+++++++++++++++++++++++++++++++++++++++++++++++++++
# calculate prediction
#+++++++++++++++++++++++++++++++++++++++++++++++++++
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
    result = EUC(data, classification, test_data, 10)
    return result

#+++++++++++++++++++++++++++++++++++++++++++++++++++
# calculate classification
#+++++++++++++++++++++++++++++++++++++++++++++++++++
def preprocess(df, column):
    average = []
    raw_data = np.array(df)
    for i in range(1, len(column)):
        temp = raw_data[:, i]
        avg = sum(temp)/len(temp)
        average.append(avg)
    data = raw_data.tolist()

    classification = []
    for D in data:
        temp = D.pop(0)
        classification.append(temp)
    data = np.array(data)
    average = np.array(average)
    return data, classification, average


def SVM(x, y, data_in):
    # X_new = SelectKBest(chi2, k=5).fit_transform(train, train_class)

    clf = svm.SVC(C=0.8, kernel='linear', gamma=1, coef0=0.8)
    clf.fit(x, y)
    result_1 = clf.predict(data_in.reshape(1, -1))
    return result_1


def classify(dic):
    df = pd.read_csv('Data.csv')
    df = df.drop(columns=['id', 'Unnamed: 32'])
    column_list = list(df)

    data, classification, average = preprocess(df, column_list)
    column_list.pop(0)
    for i in range(len(column_list)):
        if column_list[i] in dic:
            average[i] = dic[column_list[i]]
    result = SVM(data, classification, average)
    return result


if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True,use_reloader = True)