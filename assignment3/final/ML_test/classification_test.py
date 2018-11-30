import pandas as pd
import numpy as np
from sklearn import svm

def preprocess(df, column):
    average = []
    raw_data = np.array(df)
    for i in range(1, len(column)):
        temp = raw_data[:, i]
        avg = sum(temp) / len(temp)
        #print("{} {} - {}".format(column[i], min(temp), max(temp)))
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

df = pd.read_csv('Data.csv')
df = df.drop(columns=['id', 'Unnamed: 32'])
column_list = list(df)
column_list.pop(0)
data, classification, average = preprocess(df, column_list)
OUT = []
C = 0
for i in range(data.shape[0]):
    one_data = data[i]
    train_data = []
    train_class = []
    for j in range(data.shape[0]):
        if j != i:
            train_data.append(data[j].tolist())
            train_class.append(classification[j])
    clf = svm.SVC(C=0.8, kernel='linear', gamma=1, coef0=0.8)
    clf.fit(train_data, train_class)
    result_1 = clf.predict(one_data.reshape(1, -1))
    OUT.append(result_1[0])
    print(result_1)
for i in range(len(classification)):
    if OUT[i] != classification[i]:
        C += 1
print("SVM cross-validation result: ", C/len(classification))