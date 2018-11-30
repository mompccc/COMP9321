import time
import requests
import pymongo
import re
from flask import Flask, request
from flask_restplus import Resource, Api, fields, reqparse

app = Flask(__name__)
api = Api(app)

host = 'mongodb://admin:zxcv1234@ds251022.mlab.com:51022/comp9321'
client = pymongo.MongoClient(host=host)
db = client['comp9321']
COLLECTION = db['ass2']
COLLECTION_1 = db['collection_dict']

indicator_model = api.model('indicator', {"indicator_id": fields.String})
parser = reqparse.RequestParser()
parser.add_argument('query', type=str, location='args')

created_info = {
    "location": "",
    "collection_id": "",
    "creation_time": "",
    "indicator": ""
}


@api.route('/collections')
class Collections(Resource):
    @api.response(200, 'Data already exist')
    @api.response(201, 'Created')
    @api.response(400, 'Invalid indicators')
    @api.expect(indicator_model, validate=True)
    def post(self):
        entries = []
        book = request.json
        indicator_id = book['indicator_id']
        url = 'http://api.worldbank.org/v2/countries/all/indicators/{}?date=2012:2017&format=json&page=2'.format(indicator_id)
        resp = requests.get(url=url)
        data = resp.json()

        if 'message' in data[0]:
            if data[0]['message'][0]['key'] == "Invalid value":
                return {"message": "Indicator id does not exist"}, 400

        temp = COLLECTION_1.find_one({'indicator': indicator_id}, projection={'_id': False})
        print(temp)
        if temp:
            return temp, 200

        id_self = COLLECTION_1.find().count() + 1
        current_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.localtime())
        temp_info = created_info
        collection_id = "collection_{}".format(id_self)
        temp_info['location'] = "/comp9321/ass2/" + collection_id
        temp_info['collection_id'] = collection_id
        temp_info['indicator'] = indicator_id
        temp_info['creation_time'] = current_time
        COLLECTION_1.insert(temp_info)
        temp_info.pop('_id')

        for d in data[1]:
            temp_dict = {"country": d['country']['value'],
                         "date": d['date'],
                         "value": d['value']}
            entries.append(temp_dict)
        temp_collection = {"collection_id": collection_id,
                           "indicator": indicator_id,
                           "indicator_value": data[1][0]['indicator']['value'],
                           "creation_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.localtime()),
                           "entries": entries}
        COLLECTION.insert(temp_collection)
        return temp_info, 201

    @api.response(200, 'OK')
    @api.response(400, 'There is no collection')
    def get(self):
        temp_find = COLLECTION_1.find()
        count_1 = temp_find.count()
        if count_1 == 0:
            return {"message": "There is no collection"}, 400
        temp = []
        for d in temp_find:
            d.pop('_id')
            temp.append(d)
        return temp, 200


@api.route('/collections/<collection_id>')
class Operation(Resource):
    @api.response(200, 'Deleted successfully')
    @api.response(400, 'Invalid collection_id')
    def delete(self, collection_id):
        temp = COLLECTION.find_one({'collection_id': collection_id})
        if not temp:
            return {"message": "collection_id does not exist"}, 400

        COLLECTION.delete_one({'collection_id': collection_id})
        COLLECTION_1.delete_one({'collection_id': collection_id})
        return {"message": "Collection = {} is removed from the database!".format(collection_id)}, 200

    @api.response(200, 'OK')
    @api.response(400, 'Invalid collection_id')
    def get(self, collection_id):
        temp = COLLECTION.find_one({'collection_id': collection_id}, projection={'_id': False})
        if not temp:
            return {"message": "collection_id does not exist"}, 400

        return temp, 200


@api.route('/collections/<collection_id>/<year>/<country>')
class Operation1(Resource):
    @api.response(200, 'OK')
    @api.response(400, 'Error')
    def get(self, collection_id, year, country):
        query = {'collection_id': collection_id,
                 'entries.country': {'$eq': country},
                 'entries.date': {'$eq': year}}
        temp = COLLECTION.find_one(query, projection={'_id': False, 'entries.$': 1})
        temp_2 = COLLECTION.find_one({'collection_id': collection_id}, projection={'_id': False})

        if not temp:
            return {"message": "can't find relevant data"}, 400

        target_dict = {"collection_id": collection_id,
                       "indicator": temp_2['indicator'],
                       "country": country,
                       "year": year,
                       "value": temp['entries'][0]['value']}

        return target_dict, 200


@api.route('/collections/<collection_id>/<year>')
class Operation2(Resource):
    @api.response(200, 'OK')
    @api.response(400, 'Error')
    @api.expect(parser)
    def get(self, collection_id, year):
        data = COLLECTION.find_one({'collection_id': collection_id}, projection={'_id': False})
        if not data:
            return {"message": "can't find relevant data"}, 400

        entries = data['entries']
        new_entries = []
        for d in entries:
            if d['date'] == year:
                if not d['value']:
                    d['value'] = 0
                new_entries.append(d)
        new_entries.sort(key=lambda x: x["value"], reverse=True)
        for d in new_entries:
            if d['value'] == 0:
                d['value'] = None

        if not new_entries:
            return {"message": "can't find relevant data"}, 400

        result = {"indicator": data['indicator'],
                  "indicator_value": data['indicator_value'],
                  "entries": []}
        condition_dict = request.args.to_dict()
        if condition_dict:
            condition = condition_dict['query']
            head = re.findall(r'[a-zA-z]+', condition)[0]
            tail = int(re.findall(r'\d+', condition)[0])
            if head != 'top' and head != 'bottom':
                return {"message": "query must be 'top' or 'bottom'"}, 400
            if tail > 100 or tail < 1:
                return {"message": "number must be an integer value between 1 and 100"}, 400
            if head == 'top':
                result["entries"] = new_entries[:tail]
                return result, 200

            result["entries"] = new_entries[-tail:]
            return result, 200

        return new_entries, 200


if __name__ == '__main__':
    app.run(debug=True)