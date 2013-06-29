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
REG_LAMBDA = 2.5
# NUM_ITERATIONS = 400

PARAMS_FILE = 'params.pkl.gz'


def read_set():
    with gzip.open(os.path.join(os.path.pardir, os.path.join(os.path.pardir, os.path.join('dataset', 'mnist.pkl.gz')))) as f:
        return cPickle.load(f)


def sigmoid(z):
    z = np.asarray(z)
    return np.ones(z.shape)/(np.ones(z.shape) + np.exp(np.negative(z)))


def sigmoid_gradient(z):
    z = np.asarray(z)
    return sigmoid(z)*(np.ones(z.shape) - sigmoid(z))


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


def unroll_params(params, topology):
    array = []
    idx = 0
    for s in topology:
        array.append(np.array(params[idx:idx+(s[0]*s[1])]).reshape(s))
        idx = (s[0]*s[1])
    return np.array(array)


def recode_labels(num_labels, labels):
    # num_labels = len(labels)
    comp = np.arange(num_labels)
    vec_labels = np.zeros((labels.shape[0], num_labels), dtype=int)
    for i in range(vec_labels.shape[0]):
        vec_labels[i] = np.equal(labels[i], comp)
    return vec_labels


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
            deltas.insert(0, np.dot(deltas[idx], params[idx]) * sigmoid_gradient(np.insert(activations[idx-1]['z'], 0, 1, 1)))

    # calculate gradients
    params_grad = []
    for l in range(num_hidden_layers, -1, -1):
        idx = l-(num_hidden_layers+1)
        if l == num_hidden_layers:
            DELTA = np.dot(deltas[-1].T, np.insert(activations[idx-1]['a'], 0, 1, 1))
        else:
            DELTA = np.dot(deltas[idx][:,1:].T, np.insert(activations[idx-1]['a'], 0, 1, 1))

        params_grad.insert(0, (1.0/len(X)) * DELTA)

        # regularize gradients, masking out the bias term
        mask = np.insert(np.ones((params[l].shape[0],params[l].shape[1]-1)), 0, 0, 1)
        params_grad[idx] += (reg_lambda/len(X)) * (mask * params[l])

    # return a tuple containing the cost and the rolled gradients
    return J, np.concatenate([a.flatten() for a in params_grad])


def nn_run(nn_params, nn_topology, X):
    # unroll params
    params = unroll_params(nn_params, nn_topology)

    # compute forward propagation
    x = np.atleast_2d(np.insert(X, 0, 1, 1))
    for l in range(len(nn_topology)):
        z = np.dot(x, params[l].T)
        a = sigmoid(z)
        x = np.atleast_2d(np.insert(a, 0, 1, 1))
    hypothesis = a
    return hypothesis.argmax(axis=1)


def predict(img):
    thumb = pil.thumbnail((IMAGE_SIDE,IMAGE_SIDE), pil.ANTIALIAS)
    X = np.array(thumb)
    rand_params, topo = rand_initialize_weights(INPUT_LAYER_SIZE, NUM_HIDDEN_LAYERS, HIDDEN_LAYER_SIZE, NUM_LABELS)
    if os.path.exists(PARAMS_FILE):
        with gzip.open(PARAMS_FILE, 'rb') as f:
            params = cPickle.load(f)
        return nn_run(params, topo, X)
    else:
        return nn_run(rand_params, topo, X)


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
    params, topo = rand_initialize_weights(INPUT_LAYER_SIZE, NUM_HIDDEN_LAYERS, HIDDEN_LAYER_SIZE, NUM_LABELS)
    roll_params = np.concatenate([a.flatten() for a in params])


    def cost_function(PARAMS):
        return nn_cost_function(PARAMS, topo, NUM_HIDDEN_LAYERS, NUM_LABELS, train_set[0], train_set[1], REG_LAMBDA)

    t0 = time()
    print '\nLearning parameters...'
    min_params, min_J, info = fmin_l_bfgs_b(cost_function, roll_params)
    print 'Total learning time: %s' % str(timedelta(seconds=time()-t0))

    print '\nPredicting...'
    predicted = nn_run(min_params, topo, valid_set[0])

    print '\nTraining set accuracy: %.4f%%' % (np.mean(predicted == valid_set[1]) * 100)

    print '\nFinished. Saving learned parameters...'
    with gzip.open(PARAMS_FILE, 'wb') as f:
        cPickle.dump(min_params, f)

    print '\nOk.\n'

if __name__ == '__main__':
    train()