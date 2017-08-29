# coding: utf-8
"""
===============================================================================
MNIST Handwritten Digit Classifier with Neural Network
===============================================================================
Implementation of multi-layer neural network with stochastic gradient descent
(SGD) optimization method including parameters for learning rate and
regularization.

Other functions are grouped in class instead of separated into files for
readability.

Author: Pannawit Samatthiyadikun
"""

# Standard python library imports.
import array
import functools
import logging
import operator
import os
import pickle
import random
import struct
import sys
import tempfile
import time

import gzip
from urllib.parse import urljoin
from urllib.request import urlretrieve

# Related major package imports.
import numpy as np

# Application specific imports.

__author__ = "Pannawit Samatthiyadikun"
__author_email__ = "s.pannawit@gmail.com"
__docformat__ = "NumPy"


class Activator:
    """Collection class of network layers activation functions

    Four types of activations with their derivates are available:
    - Sigmoid
    - Softmax
    - Tanh
    - ReLU

    Input of any functions is 1-d vector.
    """

    @staticmethod
    def sigmoid(z):
        # Prevent overflow for exponential.
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        return Activator.sigmoid(z) * (1 - Activator.sigmoid(z))

    @staticmethod
    def softmax(z):
        # Prevent overflow for exponential.
        z = np.clip(z, -500, 500)
        return np.exp(z) / np.sum(np.exp(z))

    @staticmethod
    def softmax_prime(z):
        return Activator.softmax(z) * (1 - Activator.softmax(z))

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_prime(z):
        return 1 - Activator.tanh(z) * Activator.tanh(z)

    @staticmethod
    def relu(z):
        return np.maximum(z, 0)

    @staticmethod
    def relu_prime(z):
        return float(z > 0)


# Neural Network
# -----------------------------------------------------------------------------
class NeuralNetwork:

    def __init__(self,
                 sizes=list(),
                 learning_rate=0.1,
                 regularization=5.0,
                 mini_batch_size=256,
                 max_steps=25,
                 activator_fn=(Activator.sigmoid, Activator.sigmoid_prime)):

        # Input layer is layer 0, followed by hidden layers layer 1, 2, 3...
        self.sizes = sizes
        self.num_layers = len(sizes)

        # First term corresponds to layer 0 (input layer).
        # No weights enter the input layer and hence self.weights[0] is
        # redundant.
        # self.weights = [np.array([0])] + [np.random.randn(y, x) / np.sqrt(y)
        #                                   for x, y in zip(self.sizes[:-1], self.sizes[1:])]

        # Weights generated at the middle of sigmoid function is easier to learn
        self.weights = [np.array([0])] + [
            np.random.normal(loc=0, scale=(1 / np.sqrt(x)), size=(y, x))
            for x, y in zip(self.sizes[:-1], self.sizes[1:])]

        # Input layer does not have any biases. self.biases[0] is redundant.
        self.biases = [np.random.randn(y, 1) for y in sizes]

        # Input layer has no weights, biases associated. Hence z = wx + b is not
        # defined for input layer. self.zs[0] is redundant.
        self._zs = [np.zeros(bias.shape) for bias in self.biases]

        # Training examples can be treated as activations coming out of input
        # layer. Hence self.activations[0] = (training_example).
        self._activations = [np.zeros(bias.shape) for bias in self.biases]

        self.mini_batch_size = mini_batch_size
        self.max_steps = max_steps
        self.eta = learning_rate
        self.lmbda = regularization

        self.activator_fn = activator_fn

        self.num_steps = 0
        self.steps_cost = []
        self.steps_accuracy = []

        logging.info('neural networks is initialized')

    def fit(self, training_data, validation_data=None):
        """Fit the Neural Network on provided training data.

        Fitting is carried out using Stochastic Gradient Descent algorithm
        (SGD) with mini batches.

        Parameters
        ----------
        training_data : list of tuple
            A list of tuples of numpy arrays, ordered as (image, label).
        validation_data : list of tuple, optional
            Same as `training_data`, if provided, the network will display
            validation accuracy after each epoch.
        """
        for i_step in range(self.max_steps):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + self.mini_batch_size] for k in
                range(0, len(training_data), self.mini_batch_size)]

            for mini_batch in mini_batches:
                nabla_b = [np.zeros(bias.shape) for bias in self.biases]
                nabla_w = [np.zeros(weight.shape) for weight in self.weights]

                for x, y in mini_batch:
                    self._activations, self._zs = self._forward_prop(x)
                    delta_nabla_b, delta_nabla_w = self._back_prop(y)
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                regularization = 1 - self.eta * (self.lmbda / len(training_data))

                self.weights = [
                    regularization * w - (self.eta / len(mini_batch)) * nw
                    for w, nw in zip(self.weights, nabla_w)]
                self.biases = [
                    b - (self.eta / len(mini_batch)) * nb
                    for b, nb in zip(self.biases, nabla_b)]

            self.steps_cost.append(self.total_cost(training_data))
            self.num_steps += 1

            if validation_data:
                correct = self.validate(validation_data)
                self.steps_accuracy.append(correct / len(validation_data) * 100)
                logging.info('processed steps {0}/{1} --- '
                             'cost {2:.4f} --- '
                             'accuracy {3:.3f} % ({4}/{5})'.
                             format(self.num_steps, self.max_steps,
                                    self.steps_cost[-1],
                                    self.steps_accuracy[-1], correct,
                                    len(validation_data)))
            else:
                logging.info('processed steps {0}/{1},'.
                             format(self.num_steps, self.max_steps))

    def predict(self, x):
        a, _ = self._forward_prop(x)
        return np.argmax(a[-1])

    def validate(self, validation_data):
        return sum(int(self.predict(x) == y) for x, y in validation_data)

    def total_cost(self, data):

        def cost_fn(a, y):
            a = np.clip(a, 0, 1)
            return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

        cost = 0.0
        do_print = True
        for x, y in data:
            a, _ = self._forward_prop(x)
            if do_print:
                do_print = False
            cost += cost_fn(a[-1], vectorized_result(y)) / len(data)
        cost += 0.5 * (self.lmbda / len(data)) * sum(
            np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def _forward_prop(self, x):
        activations = [x[np.newaxis, :].T]  # list to store all the activations, layer by layer
        z = []
        for i_layer in range(1, self.num_layers):
            z.append(
                self.weights[i_layer].dot(activations[i_layer - 1]) +
                self.biases[i_layer])
            activations.append(self.activator_fn[0](z[-1]))
        zs = [np.array([0])] + z
        return activations, zs

    def _back_prop(self, y):
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        error = (self._activations[-1] - vectorized_result(y))  # version with cross entropy
        # error = (self._activations[-1] - y) * self.activator_fn[1](self._zs[-1])
        nabla_b[-1] = error
        nabla_w[-1] = error.dot(self._activations[-2].T)

        for l in range(self.num_layers - 2, 0, -1):
            error = np.multiply(
                self.weights[l + 1].T.dot(error),
                self.activator_fn[1](self._zs[l])
            )
            nabla_b[l] = error
            nabla_w[l] = error.dot(self._activations[l - 1].T)

        return nabla_b, nabla_w


class Mnist:

    def __init__(self, fname=None, url=None):
        self.fname = fname
        self.target_dir = None
        self.url = url

        def download_and_parse_files(d):
            tmp = dict()
            for key, value in d.items():
                if isinstance(value, dict):
                    tmp[key] = download_and_parse_files(value)
                else:
                    tmp[key] = self.download_and_parse_file(
                        value, target_dir=self.target_dir)
                    if key is 'images': # reshape and normalize
                        tmp[key] = tmp[key].reshape((
                            tmp[key].shape[0],
                            tmp[key].shape[1] * tmp[key].shape[2])) / 255.0
                    logging.info('{0} is loaded'.format(value))
            return tmp

        self.data = download_and_parse_files(fname)

        logging.info('MNIST dataset is initialized')

    def download_file(self, fname, target_dir=None, force=False):
        """Download fname from the url, and save it to target_dir,
        unless the file already exists, and force is False.

        Parameters
        ----------
        fname : str
            Name of the file to download
        target_dir : str
            Directory where to store the file
        force : bool
            Force downloading the file, if it already exists

        Returns
        -------
        fname : str
            Full local path of the downloaded file
        """
        if not target_dir:
            target_dir = tempfile.gettempdir()
            self.target_dir = target_dir
        target_fname = os.path.join(target_dir, fname)

        if force or not os.path.isfile(target_fname):
            url = urljoin(self.url, fname)
            urlretrieve(url, target_fname)

        return target_fname

    def parse_idx(self, fd):
        """Parse an IDX file, and return it as a numpy array.

        Parameters
        ----------
        fd : file
            File descriptor of the IDX file to parse

        Returns
        -------
        data : numpy.ndarray
            Numpy array with the dimensions and the data in the IDX file
        """
        DATA_TYPES = {0x08: 'B',  # unsigned byte
                      0x09: 'b',  # signed byte
                      0x0b: 'h',  # short (2 bytes)
                      0x0c: 'i',  # int (4 bytes)
                      0x0d: 'f',  # float (4 bytes)
                      0x0e: 'd'}  # double (8 bytes)

        header = fd.read(4)
        if len(header) != 4:
            raise ValueError('Invalid IDX file, file empty or does not contain a full header.')

        zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

        if zeros != 0:
            raise ValueError('Invalid IDX file, file must start with two zero bytes. '
                             'Found 0x%02x' % zeros)

        try:
            data_type = DATA_TYPES[data_type]
        except KeyError:
            raise ValueError('Unknown data type 0x%02x in IDX file' % data_type)

        dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                        fd.read(4 * num_dimensions))

        data = array.array(data_type, fd.read())
        data.byteswap()  # looks like array.array reads data as little endian

        expected_items = functools.reduce(operator.mul, dimension_sizes)
        if len(data) != expected_items:
            raise ValueError('IDX file has wrong number of items. '
                             'Expected: %d. Found: %d' % (expected_items, len(data)))

        return np.array(data).reshape(dimension_sizes)

    def download_and_parse_file(self, fname, target_dir=None, force=False):
        """Download the IDX file named fname from the URL specified in DATASET_URL
        and return it as a numpy array.

        Images are returned as a 3D numpy array (samples * rows * columns).
        To train machine learning models, usually a 2D array is used
        (samples * features). To get it, simply use:

        images = self.download_and_parse_file(fname)
        x = images.reshape((images.shape[0], images.shape[1] * images.shape[2]))

        Parameters
        ----------
        fname : str
            File name to download and parse
        target_dir : str
            Directory where to store the file
        force : bool
            Force downloading the file, if it already exists

        Returns
        -------
        data : numpy.ndarray
            Numpy array with the dimensions and the data in the IDX file
        """
        fname = self.download_file(fname, target_dir=target_dir, force=force)
        fopen = gzip.open if os.path.splitext(fname)[1] == '.gz' else open
        with fopen(fname, 'rb') as fd:
            return self.parse_idx(fd)


class Experiment:
    """General experiment flow

    """

    def __init__(self, model_kwargs=None,
                 model_cls=None,
                 max_experiments=5,
                 output_dir='output',
                 summary_dir='summary'):
        self.model_kwargs = model_kwargs
        self.model_cls = model_cls
        self.max_experiments = max_experiments
        self.output_dir = os.path.abspath(output_dir) + '/'
        self.models_accuracy = []

        try:
            os.makedirs(self.output_dir)
        except OSError:
            pass  # already exists

    def perform(self, training_data, validation_data):
        training_data = zip(*(training_data['images'], training_data['labels']))
        training_data = [d for d in training_data]
        validation_data = zip(*(validation_data['images'], validation_data['labels']))
        validation_data = [d for d in validation_data]

        for i_experiments in range(self.max_experiments):
            logging.info('performing model experiment {0}/{1}'.
                         format(i_experiments+1, self.max_experiments))
            model = self.model_cls(**model_kwargs)
            model.fit(training_data, validation_data=validation_data)

            self.models_accuracy.append(model.steps_accuracy[-1])

            # save model in binary
            fname = 'model.{0:02}'.format(i_experiments+1)
            path = os.path.join(self.output_dir, fname)
            with open(path, "wb") as f:
                pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
            logging.info('experimented model {0}/{1} saved to "{2}"'.
                         format(i_experiments + 1, self.max_experiments, path))

    def summarize(self):
        """ Summarize experiments

        Depends on user need---average result among different initialized
        params, models comparison, weights plot, activations plot, t-SNE, etc.
        """
        logging.info('percentage for all models with different initialized '
                     'parameters are {0} with {1} average'.
                     format(str(self.models_accuracy),
                            np.mean(self.models_accuracy)))
        np.savetxt(os.path.join(self.output_dir, 'summary.txt'), self.models_accuracy)


def vectorized_result(y):
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e


def setup_logging(config):
    """Setup global logging module with config

    Parameters
    ----------
    config : dict

    Notes
    -----
        * `Good logging practice in Python <http://goo.gl/7BbSIs>`_
    """
    filepath = os.path.abspath(config['dir']) + '/'
    if config['fname'] is None:
        filepath += time.strftime(config['format_time'])
    else:
        filepath += config['fname']
    filepath += '.log'

    log = logging.getLogger()
    log.setLevel(config['level'])
    log_format = logging.Formatter(config['format'])

    sh = logging.StreamHandler(config['stream'])
    sh.setFormatter(log_format)
    log.addHandler(sh)

    fh = logging.FileHandler(filepath)
    fh.setFormatter(log_format)
    log.addHandler(fh)

    logging.info('logging module is writing file: "{0}"'.format(filepath))


if __name__ == '__main__':

    # setup IO (logging)
    logging_conf = {
        'dir': '',
        'fname': None,
        'format': '[%(asctime)-s - %(levelname)-8s] %(module)s.%(funcName)s '
                  '--- %(message)s',
        'format_time': '%Y%m%d%H%M%S',
        'level': logging.INFO,
        'stream': sys.stdout
    }
    setup_logging(logging_conf)

    # setup dataset
    mnist_fname = {'train': {'images': 'train-images-idx3-ubyte.gz',
                             'labels': 'train-labels-idx1-ubyte.gz'},
                   'test': {'images': 't10k-images-idx3-ubyte.gz',
                            'labels': 't10k-labels-idx1-ubyte.gz'}}
    mnist_kwargs = {
        'url': 'http://yann.lecun.com/exdb/mnist/',
        'fname': mnist_fname
    }
    data = Mnist(**mnist_kwargs).data
    num_features = data['train']['images'].shape[1]

    # setup experiment design and run
    model_kwargs = {
        'sizes': [num_features, 196, 10],
        'learning_rate': 0.5,
        'regularization': 1.0,
        'mini_batch_size': 128,
        'max_steps': 50,
        'activator_fn': [Activator.sigmoid, Activator.sigmoid_prime]
    }
    experiment_kwargs = {
        'model_kwargs': model_kwargs,
        'model_cls': NeuralNetwork,
        'max_experiments': 5,
        'output_dir': 'output/sigmoid/',
    }
    experiment = Experiment(**experiment_kwargs)
    experiment.perform(data['train'], data['test'])
    experiment.summarize()
