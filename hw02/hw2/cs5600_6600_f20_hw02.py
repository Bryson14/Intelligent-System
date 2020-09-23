#!/usr/bin/python

#########################################
# module: cs5600_6600_f20_hw02.py
# Bryson Meiling
# A01503163
#########################################

import numpy as np
import pickle
from cs5600_6600_f20_hw02_data import *

# sigmoid function and its derivative.
# you'll use them in the training and fitting
# functions below.
def sigmoidf(x):
    return 1/(1 + np.exp(-x))

def sigmoidf_prime(x):
    # approximate sigmoid prime function
    return x * (1 - x)

# save() function to save the trained network to a file
def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(obj, fp)

# restore() function to restore the file
def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def build_nn_wmats(mat_dims):
    length = len(mat_dims)
    mat = []
    i = 0
    for dim in range(length - 1):
        arr = np.random.normal(0, 1, (mat_dims[i], mat_dims[i+1]))
        # mean, standard deviation, shape of matrix
        i += 1
        mat.append(arr)

    return mat


def build_231_nn():
    return build_nn_wmats((2, 3, 1))

def build_2331_nn():
    return build_nn_wmats((2, 3, 3, 1))

def build_221_nn():
    return build_nn_wmats((2, 2, 1))

def build_838_nn():
    return build_nn_wmats((8, 3, 8))

def build_949_nn():
    return build_nn_wmats((9, 4, 9))

def build_4221_nn():
    return build_nn_wmats((4, 2, 2, 1))

def build_421_nn():
    return build_nn_wmats((4, 2, 1))

def build_121_nn():
    return build_nn_wmats((1, 2, 1))

def build_1221_nn():
    return build_nn_wmats((1, 2, 2, 1))

## Training 3-layer neural net.
## X is the matrix of inputs
## y is the matrix of ground truths.
## build is a nn builder function.
def train_3_layer_nn(numIters, X, y, build):
    # iterations, input, ground truth, size
    wmat = build()
    print(f"wmat before:\n{wmat}")
    # feed forward
    for i in range(numIters):
        z2 = X.dot(wmat[0])
        a2 = sigmoidf(z2)
        z3 = a2.dot(wmat[1])
        y_hat = sigmoidf(z3)

        # backpropagation
        y_error = y - y_hat
        y_hat_delta = y_error * sigmoidf_prime(y_hat)
        a2_error = y_hat_delta.dot(wmat[1].T)
        a2_delta = a2_error * sigmoidf_prime(a2)

        # readjust weights
        wmat[1] += a2.T.dot(y_hat_delta)
        wmat[0] += X.T.dot(a2_delta)

    print(f"wmat after:\n{wmat}")
    return wmat


def train_4_layer_nn(numIters, X, y, build):
    # iterations, input, ground truth, size
    wmat = build()
    print(f"wmat before:\n{wmat}")
    # feed forward
    for i in range(numIters):
        z2 = X.dot(wmat[0])
        a2 = sigmoidf(z2)
        z3 = a2.dot(wmat[1])
        a3 = sigmoidf(z3)
        z4 = a3.dot(wmat[2])
        y_hat = sigmoidf(z4)

        # backpropagation
        y_error = y - y_hat
        y_hat_delta = y_error * sigmoidf_prime(y_hat)

        a3_error = y_hat_delta.dot(wmat[2].T)
        a3_delta = a3_error * sigmoidf_prime(a3)

        a2_error = a3_delta.dot(wmat[1].T)
        a2_delta = a2_error * sigmoidf_prime(a2)

        # readjust weights
        wmat[2] += a3.T.dot(y_hat_delta)
        wmat[1] += a2.T.dot(a3_delta)
        wmat[0] += X.T.dot(a2_delta)

    print(f"wmat after:\n{wmat}")
    return wmat

def fit_3_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
    z2 = x.dot(wmats[0])
    a2 = sigmoidf(z2)
    z3 = a2.dot(wmats[1])
    y_hat = sigmoidf(z3)

    if not thresh_flag:
        return y_hat
    else:
        y_hat = y_hat > thresh
        y_hat = y_hat * 1
        return y_hat

def fit_4_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
    z2 = x.dot(wmats[0])
    a2 = sigmoidf(z2)
    z3 = a2.dot(wmats[1])
    a3 = sigmoidf(z3)
    z3 = a3.dot(wmats[2])
    y_hat = sigmoidf(z3)

    if not thresh_flag:
        return y_hat
    else:
        y_hat = y_hat > thresh
        y_hat = y_hat * 1
        return y_hat



