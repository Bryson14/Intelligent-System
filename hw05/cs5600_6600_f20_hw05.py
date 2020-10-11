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


def define_convnet_slide22_architecture():
    input_layer = input_data(shape=[None, 28, 28, 1])
    conv_layer = conv_2d(input_layer, nb_filter=20, filter_size=5, activation="sigmoid", name="conv_layer_1")
    pool_layer = max_pool_2d(conv_layer, 2, name="pool_layer_1")
    fc_layer_1 = fully_connected(pool_layer, 100, activation="sigmoid", name="fc_layer_1")
    fc_layer_2 = fully_connected(fc_layer_1, 10, activation="softmax", name="fc_layer_2")
    network = regression(fc_layer_2, optimizer="sgd", loss="categorical_crossentropy", learning_rate=0.1)
    model = tflearn.DNN(network)

    return model

def make_tfl_convnet_slide22():
    # X is the training data set; Y is the labels for X.
    # testX is the testing data set; testY is the labels for testX.
    # loads the data
    model = define_convnet_slide22_architecture()
    return model

def load_tfl_convnet_slide22(model_path):
    model = define_convnet_slide22_architecture()
    model.load(model_path)
    return model


def define_shallow_tfl_ann():
    input_layer = input_data(shape=[None, 28, 28, 1])
    fc_layer = fully_connected(input_layer, 20, activation='sigmoid', name='fc_layer')
    output = fully_connected(fc_layer, 10, activation='softmax', name='output')
    network = regression(output, optimizer='sgd', loss="categorical_crossentropy", learning_rate=0.1)
    model = tflearn.DNN(network)

    return model

def make_shallow_tfl_ann():
    return define_shallow_tfl_ann()

def load_shallow_tfl_ann(model_path):
    model = define_shallow_tfl_ann()
    model.load(model_path)
    return model

def define_deeper_tfl_convnet():
    input_layer = input_data(shape=[None, 28, 28, 1])
    conv_layer = conv_2d(input_layer, nb_filter=20, filter_size=5, activation="sigmoid", name="conv_layer_1")
    pool_layer = max_pool_2d(conv_layer, 2, name="pool_layer_1")
    conv_layer_2 = conv_2d(pool_layer, nb_filter=40, filter_size=5, activation='relu', name="conv_layer_2")
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name="pool_layer_2")
    fc_layer_1 = fully_connected(pool_layer_2, 100, activation="relu", name="fc_layer_1")
    fc_layer_1 = dropout(fc_layer_1, 0.5)
    fc_layer_2 = fully_connected(fc_layer_1, 200, activation="relu", name="fc_layer_2")
    fc_layer_2 = dropout(fc_layer_2, 0.5)
    output = fully_connected(fc_layer_2, 10, activation='softmax', name='output')
    network = regression(output, optimizer="sgd", loss="categorical_crossentropy", learning_rate=0.1)

    model = tflearn.DNN(network)

    return model

def make_deeper_tfl_convnet():
    return define_convnet_slide22_architecture()

def load_deeper_tfl_convnet(model_path):
    model = define_convnet_slide22_architecture()
    model.load(model_path)
    return model

def fit_tfl_model(model, trainX, trainY, testX, testY, model_name, net_path, n_epoch=5, mbs=10):
    model.fit(trainX, trainY, n_epoch=n_epoch, shuffle=True, validation_set=(testX, testY), show_metric=True,
              batch_size=mbs, run_id =f"MNIST_ConvNet_{model_name}")
    path = str(Path(net_path, model_name))
    model.save(path)


def test_tfl_model(model, X, Y):
    return model.evaluate(X, Y)
        
    
        
