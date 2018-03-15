# Better TensorFlow

(Not really)

# The Graph

## Preprocessing

To preprocess, we got the MNIST pickle data (gzipped) from online and then preprocessed it using the following methods:

1. Reshape and standardize the image (`np.reshape(image / 255, (29 * 29, 1))`)
    - dividing by 255 standardizes as pixel values are grayscale 0-255
    - mnist pictures are supposed to be 29 * 29, so instead of using -1 here which would flatten, we use 29 * 29 to also assert that the image is a valid MNIST image

2. One-hot encode results (make the vector using `np.zeros(10)` and then encode using `vector[label_index] = 1`)
    - making an array of zeroes creates a vector that we can one-hot encode (i.e. every index except the label's index is 0)
    - next we can get the label index as the label int, as labels are 0-9, so `int(label)` works out the index
    - one-hot encoding is important since this is a classification problem, so by one-hot encoding instead of using a rounding method, we can access the raw value assigned/activated for each class, which we can then consider the "confidence"

3. Combine features and labels (`np.array(list(zip(X, y))))`)

4. Split into train, val, test (70%, 20%, and 10% is what we chose in the end for ratios)

5. Save as pickle (`pickle.dump((train, val, test), open('./data/mnist_preprocessed.pkl', 'wb'))`)
    - we create a tuple (train, test, val) to save everything all at once but also split into ratios
    - we use the Python pickle interface to pickle objects together and write them to a file
    - this is more memory efficient that preprocessing each time

Data can then be loaded using:

``` python
train, val, test = pickle.load(
    open('./data/mnist_preprocessed.pkl', 'rb'), encoding='latin1'
)
```

## Inspirations

Rather than arbitrarily choose a structure, we researched existing frameworks and common guidelines for defining ML frameworks. We took inspiration from the Session-Graph model that TensorFlow uses, as well as the Model system in Keras. We combined these into a Graph model, which borrows what we believe are the most convenient features from Keras and Tensorflow, and then implemented each method in Numpy using Gradient Descent. We also noted that for saving and loading models, HD5 and Protobuf were very powerful and popular. However, putting this into prod with an API, we discovered these libraries, although efficient, used up too many resources. We instead opted for NPZ (compressed numpy archives), which is convenient as the model is written in pure numpy, and is also popular in implementations. For the architecture, we tried many different hyperparameters, and were able to get steady results on with hidden units of 42 and 16 respectively without the use of convolutional, batch normalization, or pooling layers, and a batch size of 64. With further research, we discovered another numpy implementation by Michael Nielsen in his book "Neural Networks and Deep Learning." This implementation used a 30-10 network with a batch size of 32; as this proved to be more accurate, we chose this in training. In production, as we recognize the tradeoff between speed and accuracy, we compromised at a batch size of 128, which proved to be the point of diminishing returns in terms of speed gained from increasing batch size. This has worked steadily and we can easily reach accuracies of over 92% using this model with 20 epochs.

## Activation Functions

We chose to use sigmoid as this is a classification problem and sigmoid is standard/excels at these discrete, class-based approaches. Sigmoid also scales between 0 and 1, so we can consider the output the confidence.

We defined the Sigmoid function, which is 1/(1 + e ^ (-x)), using the Python math library to calculate _e_ as such:

``` python
def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))
```

For speed, we switched to Numpy's `exp` function instead:

``` python
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
```

The derivative of sigmoid is simply itself multiplied by the complement to 1, e.g. f * (1 - f), and can be represented as such:

``` python
def dsigmoid(x): return sigmoid(x) * (1 - sigmoid(x))
```

## Feeding data

For feeding data, we took inspiration from Tensorflow's `feed dict`, which can be provided to a session in the format:

``` python
{'X': x, 'y': y}
```

However, a more convenient approach is to just zip (X, y) for each training point. Thus, we defined the `feed` method and `batchify` (to get batches) methods as follows:

``` python
# feed just stores the training and validation data
def feed(self, train_data, val_data):
    self.train_data = train_data
    self.val_data = val_data

# Helper method to get batches
def batchify(self):
    random.shuffle(self.train_data) # shuffle the data around
    batches = []
    # we use the third parameter of range to chunk batch starting positions
    for start in range(0, len(self.train_data), self.batch_size):
        # append the data from start pos of length batch size
        batches.append(self.train_data[start:start + self.batch_size])
    console.info("Created %d batches." % len(batches))
    return batches
```

## Moving forward

To handle moving forward through the network, we first set the first layer to the input. Next, for each succeeding layer, we calculate the raw output as the dot product of the weights between the previous and current layer and the value of the previous layer, summed with the bias values for the current layer. We then activate this raw output to get the layer's final value by applying sigmoid.

``` python
def forward(self, X):
    self.activations[0] = X
    for layer in range(1, len(self.sizes)):
        self.bias_values[layer] = (
            self.weights[layer].dot(
                self.activations[layer - 1]) + self.biases[layer]
        )
        self.activations[layer] = sigmoid(self.bias_values[layer])
```

## Taking a step back

We then translated the mathematical approach discussed earlier into code, using the helper functions alongside numpy's excellent mathematical library to perform the necessary calculations.

``` python
def back(self, X, y, dB, dW):
    ndB = np.array([np.zeros(bias.shape) for bias in self.biases])
    ndW = np.array([np.zeros(weight.shape) for weight in self.weights])

    err = (self.activations[-1] - y) * dsigmoid(self.bias_values[-1])
    ndB[-1] = err
    ndW[-1] = err.dot(self.activations[-2].T)

    for l in list(range(len(self.sizes) - 1))[::-1]:
        err = np.multiply(
            self.weights[l + 1].T.dot(err),
            dsigmoid(self.bias_values[l])
        )
        ndB[l] = err
        ndW[l] = err.dot(self.activations[l - 1].transpose())
    dB = dB + ndB  # dB = [nb + dnb for nb, dnb in zip(dB, ndB)]
    dW = dW + ndW  # dW = [nw + dnw for nw, dnw in zip(dW, ndW)]
    return dB, dW
```

## Put it all together

Finally, we can combine these pieces into one `run` method (named after Tensorflow's `session.run`) like so:

``` python
def run(self, epochs=10):
    batches = self.batchify()
    for epoch in range(epochs):
        for batch in batches:
            #####
            # We now perform gradient descent
            dB = np.array([np.zeros(bias.shape) for bias in self.biases])
            dW = np.array([np.zeros(weight.shape)
                            for weight in self.weights])
            for X, y in batch:
                self.forward(X)
                dB, dW = self.back(X, y, dB, dW)
            # Adjust the weights
            self.weights = np.array([
                w - (self.learning_rate / self.batch_size) * dw
                for w, dw in zip(self.weights, dW)
            ])
            # Adjust the biases
            self.biases = np.array([
                b - (self.learning_rate / self.batch_size) * db
                for b, db in zip(self.biases, dB)
            ])
            #####
```

We can decorate this method to be more resourceful in several ways. Using TQDM, we can provide a progress bar for the batches similar to how Keras has it implemented in `model.fit`, and we can also log the current epoch (further using `self.epochs` to keep count of how many epochs the model has been trained for). We also keep track of time, and cross-validate if validation data is present (this is again similar to Keras's `model.fit`, which uses the `cross_validate` parameter to validate each epoch).

This brings us to the decorated `run` method:

``` python
def run(self, epochs=10):
    batches = self.batchify()
    for epoch in range(epochs):
        t0 = t()
        for batch in tqdm(batches):
            dB = np.array([np.zeros(bias.shape) for bias in self.biases])
            dW = np.array([np.zeros(weight.shape)
                            for weight in self.weights])
            for X, y in batch:
                self.forward(X)
                dB, dW = self.back(X, y, dB, dW)
            self.weights = np.array([
                w - (self.learning_rate / self.batch_size) * dw
                for w, dw in zip(self.weights, dW)
            ])
            self.biases = np.array([
                b - (self.learning_rate / self.batch_size) * db
                for b, db in zip(self.biases, dB)
            ])
        console.log("Processed %d batches in %.02f seconds." %
                    (len(batches), t() - t0))
        if self.val_data is not None:  # cannot use if self.val_data bc numpy
            console.info(
                "Accuracy: %02f" %
                (self.validate(self.val_data) / 100.0)
            )
        console.success("Processed epoch %d" % epoch)
        print("Processed epoch {0}.".format(epoch))
        self.epochs += 1
```

Note the use of `self.validate`, which is a simple method inspired by Michael Nielsen's take on classification problems; it uses Python truthiness alongside list comprehension to simply count the number of correct predictions. 

```
def validate(self, val_data):
    return sum(
        [(self.predict(X)[0] == y) for X, y in val_data])
```

Note here the use of `self.predict`, which is inspired by Keras's `predict` function. We simply move forward with an input, and then return the final activation layer.

``` python
def predict(self, X):
    self.forward(X)
    return self.activations[-1]
```

Since we need to know the label with highest confidence, we can apply numpy's `argmax` function to find the index (which is also the label since our labels are 0-9 and indices are 0-9 --> our labels are our indices) of the highest value.

``` python
def predict(self, X):
    self.forward(X)
    yhat = self.activations[-1]
    return np.argmax(yhat), yhat
```

Another improvement is to initialize random weights instead of zero-ed weights. This should bring us closer to the minima. To achieve this, we can use numpy's `randn` function with the shape of the weights (each node in the next layer must have a weight set, and there must be a weight in the weight set for each node in the previous layer).

``` python
self.weights = np.array([np.zeros(1)] + [
            np.random.randn(next_layer_size, previous_layer_size)
            for next_layer_size, previous_layer_size
            in zip(sizes[1:], sizes[:-1])
        ])
```

## Checkpoints

Instead of implementing Tensorflow's _very_ bulky checkpointer, we opted for Keras's very lightweight `save` and `load` methods. As discussed in the Inspirations section, we use NPZ archives with inspiration from Michael Nielsen.

``` python
def load(self, file='model.npz'):
    model = np.load('./models/%s' % file)
    self.epochs = int(model['epochs'])
    self.learning_rate = float(model['learning_rate'])
    self.weights = np.array(model['weights'])
    self.biases = np.array(model['biases'])
    self.batch_size = int(model['batch_size'])

def save(self, file='model.npz'):
    np.savez_compressed(
        file='./models/%s' % file,
        epochs=self.epochs,
        learning_rate=self.learning_rate,
        weights=self.weights,
        biases=self.biases,
        batch_size=self.batch_size
    )
```

For the API, we pretrained models from less than 50% to more than 90% accuracy.

The entire module is put under the "Graph" class in `better_tensorflow.py`. An example would be as follows:

``` python
import better_tensorflow as btf
net = btf.Graph((784, 30, 10), 0.004, 128)
```

# The API

The API still requires the `data` folder with the MNIST pickle file, which you can find under [Releases](./releases).

To build the API server, we needed to make sure I/O and other operations were non-blocking so that the training portion (which is an enormous synchronous call) is the only synchronous portion. To this end, Sanic was used. Sanic is an asyncio implementation of a web server in Python 3, and is extremely similar (if not identical) to Flask. It is regardedly much more stable than Flask, although Flask has more features. While deploying a Flask app requires infrastructure to support uptime such as a Node-based task queue and gunicorn, Sanic's use of async means it maintains uptime regardless of failure (it handles failure with a 500), and cutoff threads/timeouts are killed separately to the main process. Core metrics in the past have shown Sanic remains reliable for weeks under heavy load, even when used with AI models running on GPU threads. To this end, we believe Sanic is the perfect choice to deploy an API.

The basic Sanic setup requires route decorators. To avoid reiterating a basic tutorial of Sanic, we will not go into the specifics of setup, and instead discuss higher-level implementations. We use a CORS handler from the `sanic_cors` package to setup cross origin headers, which sits as a middleware. Extra methods are used to handle preflight requests so as not to clutter the main endpoints.

Furthermore, Sanic has special streaming responses that must be implemented; in this scenario, sanic_json and text are used as the main responses, and the `json` module is independently used so as to properly hash the NumPy floats (which are float32 or float64, not Python floats).

Logistically, the system-wide representation of the current file/module is referred to as `_this` (`_this = sys.modules[__name__]`); this is used due to the async nature of Sanic. To provide all threads with the same variables, we store variables in the module under `_this` (ie. `_this.net = btf.Graph(...)`), we do not implement locking, `multiprocessing.Variable`s, or data stores due to the simple fact that the only modifying operation is synchronous and thread-locking. As the API is being hosted on a DoC server, we also contacted CSG to open up an obfuscated port which is stored in an environmental variable (and not shown here).

The Sanic app layout is as follows:

```
import os
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
    ...

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.getenv('DOC_PORT'))
```

Note that the `train` endpoint supports GET, and simply trains, validates, and returns accuracy, so it can be called from a browser as well.

The data loading is not shown as it was discussed earlier in the Preprocessing section. The predict method is omitted and is instead shown below:

```
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
```

We read the data as base64, as this is a convenient and efficient format for moving images without storing them. Using the Python OpenCV2 module which is standard for image pipelines, we then decode the base64 string and read it as grayscale (the `0` in `imdecode` is the flag for grayscale). Normalization and reshaping as discussed in preprocessing is performed, and if the `pretrained` key is present then a pretrained model is selected according to specification. Then, a prediction is performed. We round the floats using string templating and then pass the result through a float constructor; although this seems "hacky", the float constructor is necessary so that Sanic can respond, as `float32` and `float64` (NumPy's defaults) are not hashable in JSON. We also use the int constructor as NumPy likely used uint8 to store the label.

As such, the endpoints are then available to be referenced by a web application.