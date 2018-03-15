import os
import time
import sys
_this = sys.modules[__name__]

import wget
import gzip
import numpy as np

import better_tensorflow as btf
pretrained = btf.Graph([784, 30, 10], 0.5, 128)
pretrained.load()
net = btf.Graph([784, 30, 10], 0.5, 128)
_this.pretrained = pretrained
_this.net = net

import pickle
from console_logging.console import Console
console = Console()

_this.data, _this.val_data, _this.test_data = pickle.load(
    open('./data/mnist_preprocessed.pkl', 'rb'), encoding='latin1'
)

import base64
import re
import cv2
import numpy as np

from sanic import Sanic
from sanic.response import text, json as sanic_json
import json
from sanic_cors import CORS, cross_origin
app = Sanic()
CORS(app)

import random


@app.route('/predict', methods=['OPTIONS'])
async def preflight(q):
    return text('OK', status=200)


@app.route('/reset', methods=['GET', 'POST', 'OPTIONS'])
async def reset(q):
    _this.net = btf.Graph([784, 30, 10], 0.5)
    return text('OK', status=200)


@app.route('/pretrain', methods=['GET', 'POST', 'OPTIONS'])
async def pretrain(q):
    _this.net = _this.pretrained
    return text('OK', status=200)


@app.route('/train', methods=['OPTIONS'])
async def preflight2(q):
    return text('OK', status=200)


@app.route('/train', methods=['POST', 'GET'])
async def train(q):
    epochs = q.json.get('epochs') if q.json else 1
    random.shuffle(_this.data)
    _this.net.feed(_this.data, None)
    _this.net.run(epochs)
    random.shuffle(_this.val_data)
    acc = _this.net.validate(_this.val_data[:100])
    d = {
        'accuracy': int(acc)
    }
    return sanic_json(d, dumps=json.dumps)


@app.route('/predict', methods=['POST'])
async def predict(q):
    imageURI = base64.urlsafe_b64decode(
        re.sub('^data:image/.+;base64,', '', q.json.get('image')))
    image = cv2.imdecode(np.fromstring(imageURI, dtype=np.uint8), 0)
    image = image / 255
    print(image.shape)
    _in = np.reshape(image, (784, 1))
    try:
        if q.json.get('pretrained') > 0:
            console.log("Using model-v%d.npz" % q.json.get('pretrained'))
            _this.pretrained.load(file='model-v%d.npz' %
                                  q.json.get('pretrained'))
            prediction, predictions = _this.pretrained.predict(_in)
    except Exception as e:
        prediction, predictions = _this.net.predict(_in)
    d = {
        'label': int(prediction),
        'predictions': [
            {'label': int(i), 'confidence': float('%02f' % c[0])}
            for i, c in enumerate(predictions)
        ]
    }
    j = json.dumps(d)
    print(prediction)
    print(j)
    print(d)
    return sanic_json(d, dumps=json.dumps)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7022)
