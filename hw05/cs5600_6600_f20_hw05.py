#/usr/bin/python

###########################################
# module: cs5600_6600_f20_hw05.py
# Bryson Meiling
# A01503163
###########################################

import numpy as np
import tensorflow as tf
from pathlib import Path
import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist


def define_convnet_slide22_architecture(load=False):
    input_layer = input_data(shape=[None, 28, 28, 1])
    conv_layer = conv_2d(input_layer, nb_filter=20, filter_size=5, activation="sigmoid", name="conv_layer_1")
    pool_layer = max_pool_2d(conv_layer, 2, name="pool_layer_1")
    fc_layer_1 = fully_connected(pool_layer, 100, activation="sigmoid", name="fc_layer_1")
    fc_layer_2 = fully_connected(fc_layer_1, 10, activation="softmax", name="fc_layer_2")
    network = regression(fc_layer_2, optimizer="sgd", loss="categorical_crossentropy", learning_rate=0.1)

    if not load:
        model = tflearn.DNN(network)
    else:
        model = tflearn.DNN(fc_layer_2)

    return model

def make_tfl_convnet_slide22():
    # X is the training data set; Y is the labels for X.
    # testX is the testing data set; testY is the labels for testX.
    # loads the data
    X, Y, testX, testY = mnist.load_data(one_hot=True)
    # shuffles the list knowing that each index in both list are correlated to the respective index of the other list
    p = np.random.permutation(len(X))
    X, Y = X[p], Y[p]
    del p
    p2 = np.random.permutation(len(testX))
    testX, testY = testX[p2], testY[p2]
    del p2

    # splits X, Y into training and validation data.
    trainX = X[0:50000]
    trainY = Y[0:50000]
    validX = X[50000:]
    validY = Y[50000:]
    trainX = trainX.reshape([-1, 28, 28, 1])
    testX = testX.reshape([-1, 28, 28, 1])
    validX = validX.reshape([-1, 28, 28, 1])

    model = define_convnet_slide22_architecture()
    model.fit(trainX, trainY, n_epoch=1, shuffle=True, validation_set=(testX, testY), show_metric=True,
              batch_size=10, run_id ="MNIST_ConvNet_1")
    model.save("nets/ConvNet_Slide22.tfl")
    return model

def load_tfl_convnet_slide22(model_path):
    path = Path(model_path)
    model = define_convnet_slide22_architecture(load=True)
    model.load(model_path)
    return model

def make_shallow_tfl_ann():
    # your code here
    pass

def make_deeper_tfl_convnet():
    # your code here
    pass

def load_deeper_tfl_convnet(model_path):
    # your code here
    pass

def load_shallow_tfl_ann(model_path):
    # your code here
    pass

def fit_tfl_model(model, trainX, trainY, testX, testY, model_name, net_path, n_epoch=1, mbs=10):
    # your code here
    pass

def test_tfl_model(model, X, Y):
    # your code here
    pass
        
    
        
