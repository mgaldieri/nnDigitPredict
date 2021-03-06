#! /usr/bin/env python
__author__ = 'mgaldieri'
# from django.conf import settings
from scipy.optimize import fmin_l_bfgs_b
from time import time
from datetime import timedelta
import numpy as np
import Image as pil
import cPickle, gzip, os, math


#
# Constants
#
IMAGE_SIDE = 28
INPUT_LAYER_SIZE = IMAGE_SIDE**2
NUM_HIDDEN_LAYERS = 1
HIDDEN_LAYER_SIZE = 25
NUM_LABELS = 10
REG_LAMBDA = 1.7
NUM_TRAIN_SAMPLES = 10000

PARAMS_FILE = 'params.pkl.gz'
CONFIG_FILE = 'params.cfg.gz'

CURRENT_ITERATION = 0
ACCURACY = 0.0

#
# Helper methods
#


def read_set():
    with gzip.open(os.path.join(os.path.pardir, os.path.join(os.path.pardir, os.path.join('dataset', 'mnist.pkl.gz')))) as f:
        return cPickle.load(f)


def sub_set(set):
    if NUM_TRAIN_SAMPLES == 'all':
        return set[0], set[1]
    else:
        idx = np.random.choice(set[0].shape[0], NUM_TRAIN_SAMPLES, replace=False)
        return set[0][idx,:], set[1][idx]


def rand_initialize_weights(input_layer_size, num_hidden_layers, hidden_layer_size, num_labels):
    weights = []
    topology = []
    for i in range(num_hidden_layers + 1):
        if i < num_hidden_layers:
            if i == 0:
                epsilon = math.sqrt(6)/math.sqrt(input_layer_size + hidden_layer_size)
                weights.append(np.random.uniform(-epsilon, epsilon, (hidden_layer_size, input_layer_size + 1))) #(input_layer_size + 1, hidden_layer_size))) #
            else:
                epsilon = math.sqrt(6)/math.sqrt(hidden_layer_size * 2)
                weights.append(np.random.uniform(-epsilon, epsilon, (hidden_layer_size, hidden_layer_size + 1))) #(hidden_layer_size + 1, hidden_layer_size))) #
        else:
            epsilon = math.sqrt(6)/math.sqrt(hidden_layer_size + num_labels)
            weights.append(np.random.uniform(-epsilon, epsilon, (num_labels, hidden_layer_size + 1))) #(hidden_layer_size + 1, num_labels))) #
        topology.append(weights[i].shape)
    return np.asarray(weights), topology


def recode_labels(num_labels, labels):
    # num_labels = len(labels)
    comp = np.arange(num_labels)
    vec_labels = np.zeros((labels.shape[0], num_labels), dtype=int)
    for i in range(vec_labels.shape[0]):
        vec_labels[i] = np.equal(labels[i], comp)
    return vec_labels


def gen_config():
    return {'image_side': IMAGE_SIDE,
            'input_layer_size': INPUT_LAYER_SIZE,
            'num_hidden_layers': NUM_HIDDEN_LAYERS,
            'hidden_layer_size': HIDDEN_LAYER_SIZE,
            'num_labels': NUM_LABELS,
            'reg_lambda': REG_LAMBDA,
            'num_train_samples': NUM_TRAIN_SAMPLES,
            'accuracy': ACCURACY}


def load_config(config=None):
    if config:
        global IMAGE_SIDE, INPUT_LAYER_SIZE, NUM_HIDDEN_LAYERS, HIDDEN_LAYER_SIZE, NUM_LABELS, REG_LAMBDA, NUM_TRAIN_SAMPLES, ACCURACY
        IMAGE_SIDE = config['image_side']
        INPUT_LAYER_SIZE = config['input_layer_size']
        NUM_HIDDEN_LAYERS = config['num_hidden_layers']
        HIDDEN_LAYER_SIZE = config['hidden_layer_size']
        NUM_LABELS = config['num_labels']
        REG_LAMBDA = config['reg_lambda']
        NUM_TRAIN_SAMPLES = config['num_train_samples']
        ACCURACY = config['accuracy']


def update_stdout(xk):
    global CURRENT_ITERATION
    CURRENT_ITERATION += 1
    print 'Current iteration: '+str(CURRENT_ITERATION)+'               \r',


#
# Neural network methods
#


def sigmoid(z):
    z = np.asarray(z)
    return np.ones(z.shape)/(np.ones(z.shape) + np.exp(np.negative(z)))


def sigmoid_gradient(z):
    z = np.asarray(z)
    return sigmoid(z)*(np.ones(z.shape) - sigmoid(z))


def unroll_params(params, topology):
    array = []
    idx = 0
    for s in topology:
        array.append(np.array(params[idx:idx+(s[0]*s[1])]).reshape(s))
        idx = (s[0]*s[1])
    return np.array(array)


def nn_cost_function(nn_params, nn_topology, num_hidden_layers, num_labels, X, y, reg_lambda):
    # unroll parameters
    params = unroll_params(nn_params, nn_topology)

    # compute forward propagation
    activations = [{'a': X, 'z': None}]
    x = np.atleast_2d(np.insert(X, 0, 1, 1))
    for l in range(num_hidden_layers + 1):
        z = np.dot(x, params[l].T)
        a = sigmoid(z)
        activations.append({'a': a, 'z': z})
        x = np.atleast_2d(np.insert(a, 0, 1, 1))
    hypothesis = activations[-1]['a']

    # compute error
    y_vec = recode_labels(num_labels, y)
    error = []
    for b in range(len(X)):
        e = (np.dot(np.negative(y_vec[b]), np.log(hypothesis[b].T))) - (np.dot((1 - y_vec[b]), (np.log(1 - hypothesis[b].T))))
        error.append(e)
    J = np.mean(error)

    # compute the regularization term
    sum_vec = []
    for a in params:
        sum_vec.append(np.sum(np.square(a[:,1:])))
    reg = (float(reg_lambda)/(2*len(X))) * sum(sum_vec)
    J += reg

    # backpropagation
    # calculate deltas
    deltas = []
    for l in range(num_hidden_layers+1, 0, -1):
        idx = l-(num_hidden_layers+1)
        if l == (num_hidden_layers+1):
            deltas.insert(0, hypothesis - y_vec)
        else:
            delta = np.dot(deltas[idx], params[idx]) * sigmoid_gradient(np.insert(activations[idx-1]['z'], 0, 1, 1))
            deltas.insert(0, delta[:,1:])

    # calculate gradients
    params_grad = []
    for l in range(num_hidden_layers, -1, -1):
        idx = l-(num_hidden_layers+1)
        if l == num_hidden_layers:
            DELTA = np.dot(deltas[-1].T, np.insert(activations[idx-1]['a'], 0, 1, 1))
        else:
            DELTA = np.dot(deltas[idx].T, np.insert(activations[idx-1]['a'], 0, 1, 1))

        params_grad.insert(0, (1.0/len(X)) * DELTA)

        # regularize gradients, masking out the bias term
        mask = np.insert(np.ones((params[l].shape[0],params[l].shape[1]-1)), 0, 0, 1)
        params_grad[idx] += (reg_lambda/len(X)) * (mask * params[l])

    # return a tuple containing the cost and the rolled gradients
    return J, np.concatenate([a.flatten() for a in params_grad])


#
# Run methods
#


def nn_run(nn_params, nn_topology, X):
    # unroll params
    params = unroll_params(nn_params, nn_topology)
    # compute forward propagation
    x = np.insert(np.atleast_2d(X), 0, 1, 1)
    for l in range(len(nn_topology)):
        z = np.dot(x, params[l].T)
        a = sigmoid(z)
        x = np.atleast_2d(np.insert(a, 0, 1, 1))
    hypothesis = a.argmax(axis=1)
    return hypothesis if len(hypothesis) > 1 else hypothesis[0]


def predict(img):
    img.thumbnail((IMAGE_SIDE,IMAGE_SIDE), pil.ANTIALIAS)
    X = np.array(img.convert('F'))
    X /= 255.0
    X = np.concatenate([a.flatten() for a in X]) #np.array(img.convert('L'))])
    try:
        with gzip.open(os.path.join(os.path.dirname(__file__), PARAMS_FILE), 'rb') as f:
            params = cPickle.load(f)
        try:
            with gzip.open(os.path.join(os.path.dirname(__file__), CONFIG_FILE), 'rb') as f:
                config = cPickle.load(f)
                load_config(config)
        except IOError:
            raise Exception('Error loading configurations file...')
        rand_params, topo = rand_initialize_weights(INPUT_LAYER_SIZE, NUM_HIDDEN_LAYERS, HIDDEN_LAYER_SIZE, NUM_LABELS)
        return nn_run(np.concatenate([a.flatten() for a in params]), topo, X)
    except IOError:
        rand_params, topo = rand_initialize_weights(INPUT_LAYER_SIZE, NUM_HIDDEN_LAYERS, HIDDEN_LAYER_SIZE, NUM_LABELS)
        return nn_run(np.concatenate([a.flatten() for a in rand_params]), topo, X)

#
# Main entry point
#


def train():
    print '\nReading training sets...'
    # reminder: set[image=0/label=1][sample]
    train_set, valid_set, test_set = read_set()

    print '\nInitializing neural network with:'
    print '- Input layer size:', INPUT_LAYER_SIZE
    print '- Number of hidden layers:', NUM_HIDDEN_LAYERS
    print '- Hidden layer(s) size:', HIDDEN_LAYER_SIZE
    print '- Number of output classes:', NUM_LABELS
    print '- Regularization lambda:', REG_LAMBDA
    print '- Number of training samples: ', NUM_TRAIN_SAMPLES
    params, topo = rand_initialize_weights(INPUT_LAYER_SIZE, NUM_HIDDEN_LAYERS, HIDDEN_LAYER_SIZE, NUM_LABELS)
    roll_params = np.concatenate([a.flatten() for a in params])

    sub_train_imgs, sub_train_labels = sub_set(train_set)

    def cost_function(PARAMS):
        return nn_cost_function(PARAMS, topo, NUM_HIDDEN_LAYERS, NUM_LABELS, sub_train_imgs, sub_train_labels, REG_LAMBDA)
    
    t0 = time()
    print '\nLearning parameters...'
    min_params, min_J, info = fmin_l_bfgs_b(cost_function, roll_params, callback=update_stdout)
    print 'Total number of iterations: %d' % info['nit']
    print 'Total learning time: %s' % str(timedelta(seconds=time()-t0))

    print '\nChecking accuracy...'
    predicted = nn_run(min_params, topo, valid_set[0])
    global ACCURACY
    ACCURACY = np.mean(predicted == valid_set[1]) * 100
    print 'Training set accuracy: %.4f%%' % ACCURACY

    print '\nFinished. Saving learned parameters...'
    with gzip.open(os.path.join(os.path.dirname(__file__), PARAMS_FILE), 'wb') as f:
        cPickle.dump(min_params, f)
    with gzip.open(os.path.join(os.path.dirname(__file__), CONFIG_FILE), 'wb') as f:
        cPickle.dump(gen_config(), f)
    print '\nOk.\n'


def debug_imgs():
    MOSAIC_SIDE = 10
    global NUM_TRAIN_SAMPLES
    NUM_TRAIN_SAMPLES = MOSAIC_SIDE ** 2

    train, cross, test = read_set()
    sub_samples, sub_labels = sub_set(train)

    mosaic = pil.new('F', (MOSAIC_SIDE*IMAGE_SIDE, MOSAIC_SIDE*IMAGE_SIDE))
    for i in range(MOSAIC_SIDE):
        for j in range(MOSAIC_SIDE):
            img = pil.fromarray(sub_samples[(i*MOSAIC_SIDE)+j,:].reshape((28,28))*255.0)
            offset = i*IMAGE_SIDE, j*IMAGE_SIDE
            mosaic.paste(img, offset)
    # mosaic.show()
    mosaic.convert('L').save('samples.png')

if __name__ == '__main__':
    train()
    # debug_imgs()