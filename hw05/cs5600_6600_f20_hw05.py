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
    model = define_convnet_slide22_architecture()
    return model

def load_tfl_convnet_slide22(model_path):
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
    model.fit(trainX, trainY, n_epoch=n_epoch, shuffle=True, validation_set=(testX, testY), show_metric=True,
              batch_size=mbs, run_id ="MNIST_ConvNet_1")
    model.save(Path.joinpath(net_path, model_name))


def test_tfl_model(model, X, Y):
    return model.evaluate(X, Y)
        
    
        
