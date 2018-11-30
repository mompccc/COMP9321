import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.feature_selection import SelectKBest, chi2


def preprocess(df, column):
    average = []
    raw_data = np.array(df)
    for i in range(1, len(column)):
        temp = raw_data[:, i]
        avg = sum(temp)/len(temp)
        print(column[i], max(temp), min(temp))
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


D = {'area_mean': 566.3, 'area_se': 23.56, 'perimeter_mean': 87.46, 'perimeter_worst': 99.7, 'area_worst': 711.2}
#D = {'area_mean': 1001, 'area_se': 153.4, 'perimeter_mean': 122.8, 'perimeter_worst': 184.6, 'area_worst': 2019}
result = classify(D)
print(result)
