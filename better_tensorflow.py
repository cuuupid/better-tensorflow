import os
import numpy as np
import random
from tqdm import tqdm
from time import time as t
from console_logging.console import Console
console = Console()


def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Graph(object):

    # Replicating Tensorflow's feed_dict as instead just providing train and val
    # This is a sort of morph between tf feed_dict and keras model.fit

    def feed(self, train_data, val_data):
        self.train_data = train_data
        self.val_data = val_data

    # Helper method to get batches
    def batchify(self):
        random.shuffle(self.train_data)
        batches = []
        # we use the third parameter of range to chunk batch starting positions
        for start in range(0, len(self.train_data), self.batch_size):
            # append the data from start pos of length batch size
            batches.append(self.train_data[start:start + self.batch_size])
        console.info("Created %d batches." % len(batches))
        return batches

    # Replicating session.run except using it on Graph directly
    # TODO: modularize graph, 1 to many for sessions to graphs

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

    # Helper functions to run through validation and prediction

    def validate(self, val_data):
        return sum(
            [(self.predict(X)[0] == y) for X, y in val_data])

    def predict(self, X):
        self.forward(X)
        yhat = self.activations[-1]
        return np.argmax(yhat), yhat

    # Run on a set of features, and then evaluate performance and perform GD
    # TODO: add stochastic gradient descent

    def forward(self, X):
        self.activations[0] = X
        for layer in range(1, len(self.sizes)):
            self.bias_values[layer] = (
                self.weights[layer].dot(
                    self.activations[layer - 1]) + self.biases[layer]
            )
            self.activations[layer] = sigmoid(self.bias_values[layer])

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

    # We replicate Keras for saving and loading model:
    '''
    model.load()
    model.save()
    '''

    def load(self, file='model.npz'):
        model = np.load('./models/%s' % file)
        self.epochs = int(model['epochs'])
        self.learning_rate = float(model['learning_rate'])
        self.weights = np.array(model['weights'])
        self.biases = np.array(model['biases'])
        self.batch_size = int(model['batch_size'])

        self.sizes = np.array([b.shape[0] for b in self.biases])
        self.bias_values = np.array([np.zeros(bias.shape)
                                     for bias in self.biases])
        self.activations = np.array([np.zeros(bias.shape)
                                     for bias in self.biases])

    def save(self, file='model.npz'):
        np.savez_compressed(
            file='./models/%s' % file,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weights=self.weights,
            biases=self.biases,
            batch_size=self.batch_size
        )

    # Constructor to randomize/nullify all values

    def __init__(self, sizes, learning_rate, batch_size):
        self.sizes = sizes
        self.weights = np.array([np.zeros(1)] + [
            np.random.randn(next_layer_size, previous_layer_size)
            for next_layer_size, previous_layer_size
            in zip(sizes[1:], sizes[:-1])
        ])
        self.biases = np.array([np.random.randn(layer_size, 1)
                                for layer_size in sizes])
        self.bias_values = np.array([np.zeros(bias.shape)
                                     for bias in self.biases])
        self.activations = np.array([np.zeros(bias.shape)
                                     for bias in self.biases])
        self.batch_size = batch_size
        self.epochs = 0
        self.learning_rate = learning_rate
        self.train_data = None
        self.val_data = None
        console.info("Initialized new model.")
