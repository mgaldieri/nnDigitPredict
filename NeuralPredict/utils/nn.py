__author__ = 'mgaldieri'
# from django.conf import settings
# from django.http import HttpResponse
# from scipy.optimize import minimize
import numpy as np
import Image as im
import cPickle, gzip, os, math


def read_set():
    with gzip.open(os.path.join(os.path.pardir, os.path.join(os.path.pardir, os.path.join('dataset', 'mnist.pkl.gz')))) as f:
        return cPickle.load(f)
    # sample = 435
    # print train_set[1][sample]
    # img = im.fromarray(train_set[0][sample].reshape(28, 28), 'I')
    # img.show()


def sigmoid(z):
    z = np.asarray(z)
    return np.ones(z.shape)/(np.ones(z.shape) + np.exp(np.negative(z)))


def sigmoid_gradient(z):
    z = np.asarray(z)
    return sigmoid(z)*(np.ones(z.shape) - sigmoid(z))


def rand_initialize_weights(input_layer_size, num_hidden_layers, hidden_layer_size, num_labels):
    weights = []
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
    return np.asarray(weights)


def recode_labels(num_labels, labels):
    # num_labels = len(labels)
    comp = np.arange(num_labels)
    vec_labels = np.zeros((labels.shape[0], num_labels), dtype=int)
    for i in range(vec_labels.shape[0]):
        vec_labels[i] = np.equal(labels[i], comp)
    return vec_labels


def nn_cost_function(nn_params, input_layer_size, num_hidden_layers, hidden_layer_size, num_labels, X, y, reg_lambda):
    # compute forward propagation
    activations = []
    x = np.atleast_2d(np.insert(X, 0, 1, 1))
    for l in range(num_hidden_layers + 1):
        z = np.dot(x, nn_params[l].T)
        a = sigmoid(z)
        print 'x:\n', x
        print 'theta:\n', nn_params[l]
        print 'z:\n', z
        print 'a:\n', a
        print '\n'
        activations.append({'a': a, 'z': z})
        x = np.atleast_2d(np.insert(a, 0, 1, 1))
    print '\n'
    print 'Activations:\n', activations, '\n'
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
    for a in nn_params:
        sum_vec.append(np.sum(np.square(a[:,1:])))
    reg = (float(reg_lambda)/(2*len(X))) * sum(sum_vec)
    J += reg
    # print 'Cost:\n', J, '\n'

    # backpropagation
    print 'Hypothesis:\n', hypothesis, '\n'

    deltas = []
    params_grad = []
    for l in range(num_hidden_layers + 1):
        if l == 0:
            deltas.append(hypothesis - y_vec)
            DELTA = np.dot(deltas[-1], activations[-(l+1)]['a'].T)
        else:
            # print nn_params[-l], deltas[-l].shape, sigmoid_gradient(np.insert(activations[-(l+1)]['z'], 0, 1, 1)).shape
            deltas.append(np.dot(nn_params[l].T, deltas[-l]).T * sigmoid_gradient(np.insert(activations[-(l+1)]['z'], 0, 1, 1)))
            DELTA = np.dot(deltas[-l][:,1:], activations[-(l+1)]['a'].T)
        params_grad.append((1.0/len(X)) * DELTA)

        print 'TESTE:\n', params_grad, '\n'
    # print 'y:\n', y, '\n'
    # print 'y_vec:\n', y_vec, '\n'
    # print 'deltas:\n', deltas, '\n'

#
# Constants
#
IMAGE_SIDE = 1
INPUT_LAYER_SIZE = IMAGE_SIDE**2
NUM_HIDDEN_LAYERS = 1
HIDDEN_LAYER_SIZE = 3
NUM_LABELS = 2
NUM_ITERATIONS = 400

# reminder: set[image=0/label=1][sample]
train_set, valid_set, test_set = read_set()

# print np.atleast_2d([1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,19,0,1,2,3,4,5,6,7,8,9,0])
# print np.array([[1],[0]]).shape
params = rand_initialize_weights(INPUT_LAYER_SIZE, NUM_HIDDEN_LAYERS, HIDDEN_LAYER_SIZE, NUM_LABELS)
nn_cost_function(params, INPUT_LAYER_SIZE, NUM_HIDDEN_LAYERS, HIDDEN_LAYER_SIZE, NUM_LABELS, np.atleast_2d([[0.123]]), np.asarray([1, 0], dtype=int), 1)
# print np.square(params)
# print np.asmatrix(np.square(params)).cumsum()
# print train_set[1][443]
# print recode_labels(NUM_LABELS, train_set[1])[443]