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

# The API

The API still requires the `data` folder with the MNIST pickle file, which you can find under [Releases](./releases).