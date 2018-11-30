from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np
from sklearn import svm
import math
from flask_restplus import Resource, Api, fields, reqparse

app = Flask(__name__)
api = Api(app, description="In prediction model, you must input a feature from area_mean, area_se, area_worst, "
                           "perimeter_mean and perimeter_worst")

parser = reqparse.RequestParser()
parser.add_argument('Feature to predict', type=str, location='args')
feature_model = api.model('classification', {"area_mean": fields.Float,
                                             "area_se": fields.Float,
                                             "area_worst": fields.Float,
                                             "perimeter_mean": fields.Float,
                                             "perimeter_worst": fields.Float})


@api.route('/Classification')
class ClassificationRun(Resource):
    @api.response(200, 'Classification Success')
    @api.expect(feature_model, validate=True)
    def post(self):
        values = request.json
        result = classify(values)[0]
        if result == "M":
            result = "malignant"
        else:
            result = "benign"
        return {"Classification result": result}, 200


@api.route('/Prediction')
class PredictionRun(Resource):
    @api.response(200, 'Prediction Success')
    @api.response(404, 'Feature error')
    @api.expect(feature_model, parser, validate=True)
    def post(self):
        values = request.json
        if not request.args:
            return {"message": "The feature must be one of five above."}, 400
        feature_name = request.args.to_dict()['Feature to predict']
        if feature_name not in values:
            return {"message": "The feature must be one of five above."}, 400
        values['diagnosis'] = "M"
        result = prediction(values, feature_name)
        return {"Prediction result of " + feature_name: result}, 200


def preprocess(df, column):
    average = []
    raw_data = np.array(df)
    for i in range(1, len(column)):
        temp = raw_data[:, i]
        avg = sum(temp) / len(temp)
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


def EUC(train_data, train_class, test_data, k):
    instances = train_data.shape[0]
    minus = np.tile(test_data, (instances, 1)) - train_data  # step 1: minus
    squared_m = minus ** 2  # step 2: square
    squared_dist = squared_m.sum(axis=1)  # step 3: sum up
    distance = squared_dist ** 0.5  # step 4: extraction of a root
    sorted_index = np.argsort(distance)  # step 5: sort the distance in ascending and return rank list
    all_price = []  # list to store all nearest k prices
    denominator = 0
    mu = sum(distance) / len(distance)
    for i in range(k):
        ratio = (1 / mu * (math.pi * 2) ** 0.5) * math.e ** (-(distance[sorted_index[i]]) ** 2 / (
                2 * mu ** 2))  # compute the ratio of current point distance and basement point distance
        temp_price = train_class[sorted_index[i]] * ratio  # plus weighted price
        all_price.append(temp_price)
        denominator += ratio
    return sum(all_price) / denominator


def preprocess_prediction(df, column, index):
    average = []
    raw_data = np.array(df)
    for i in range(1, len(column)):
        if i != index:
            temp = raw_data[:, i]
            average.append(sum(temp) / len(temp))
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


if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)
