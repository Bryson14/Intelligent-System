########################################################
# module: load_nets.py
# Bryson Meiling
# A01503163
# descrption: starter code for loading your project 1 nets.
########################################################

import pickle
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

### ======================= ANNs ===========================

def load_ann_audio_model_buzz1(model_path):
    """
    This networked ended training with an accuracy of around 0.55 and an accuracy of 75 %. However, when testing
    this again from a loaded state, the accuracy was 50.2 %. This is barely half.
    :param model_path: path to "ann_audio_model_buzz1.tfl"
    :return: TFLEARN deep network model
    """
    input_layer = input_data(shape=[None, 4000, 1, 1])
    fc_layer_1 = fully_connected(input_layer, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 64,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_3 = fully_connected(fc_layer_2, 3,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_3)
    model.load(model_path)
    return model

def load_ann_audio_model_buzz2(model_path):
    """
    This networked ended training with an accuracy of around 0.68 and an accuracy of 75 %. However, when testing
    this again from a loaded state, the accuracy was 32.4 %.
    :param model_path: path to "ann_audio_model_buzz2.tfl"
    :return: TFLEARN deep network model
    """
    input_layer = input_data(shape=[None, 4000, 1, 1])
    fc_layer_1 = fully_connected(input_layer, 512,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 256,
                                 activation='relu',
                                 name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 128,
                                 activation='relu',
                                 name='fc_layer_3')
    fc_layer_4 = fully_connected(fc_layer_3, 3,
                                 activation='softmax',
                                 name='fc_layer_4')
    model = tflearn.DNN(fc_layer_4)
    model.load(model_path)
    return model

def load_ann_audio_model_buzz3(model_path):
    """
    This networked ended training with an accuracy of around 0.55 and an accuracy of 80 % This was the better
    performing of the audio ANNs. Originally i tried and failed to make a deep 4 hidden layer model, but its
    performance was terrible around 40 %. Maybe it had better potencial, but i didn't have the time to train this
    for months. On validation it had an accuracy of 56.2 %.
    This ass the best ann audio net.
    :param model_path: path to "ann_audio_model_buzz3.tfl"
    :return: TFLEARN deep network model
    """
    input_layer = input_data(shape=[None, 4000, 1, 1])
    fc_layer_1 = fully_connected(input_layer, 64,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 64,
                                 activation='sigmoid',
                                 name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 3,
                                 activation='softmax',
                                 name='fc_layer_3')
    model = tflearn.DNN(fc_layer_3)
    model.load(model_path)
    return model

def load_ann_image_model_bee1_gray(model_path):
    """
    This model was trained for 15 epochs at a batch size of 8 and learning rate of 0.01
    Accuracy was 95% during training and 0.23 cost. Validationg was 91.69 %
    This was the best ann image net
    :param model_path: path to "ann_image_model_bee1_gray.tfl"
    :return: TFLEARN deep network model
    """
    input_layer = input_data(shape=[None, 64, 64, 1])
    fc_layer_1 = fully_connected(input_layer, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model

def load_ann_image_model_bee2_1s_gray(model_path):
    """
    This model was trained for 20 epochs at a batch size of 8 and learning rate of 0.01
    Accuracy was 94% during training and 0.23 cost. Validation was 75.68 %
    :param model_path: path to "ann_image_model_bee2_1s_gray.tfl"
    :return: TFLEARN deep network model
    """
    input_layer = input_data(shape=[None, 64, 64, 1])
    fc_layer_1 = fully_connected(input_layer, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 128,
                                 activation='softmax',
                                 name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 128,
                                 activation='softmax',
                                 name='fc_layer_3')
    fc_layer_4 = fully_connected(fc_layer_3, 2,
                                 activation='relu',
                                 name='fc_layer_4')
    model = tflearn.DNN(fc_layer_4)
    model.load(model_path)
    return model

def load_ann_image_model_bee4_gray(model_path):
    """
    This model was trained for 40 epochs at a batch size of 8 and learning rate of 0.01
    Accuracy was 61 % during training and .678 cost. Validation was 55.2 %
    :param model_path: path to "ann_image_model_bee4_gray.tfl"
    :return: TFLEARN deep network model
    """
    input_layer = input_data(shape=[None, 64, 64, 1])
    fc_layer_1 = fully_connected(input_layer, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 128,
                                 activation='softmax',
                                 name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 128,
                                 activation='softmax',
                                 name='fc_layer_3')
    fc_layer_4 = fully_connected(fc_layer_3, 64,
                                 activation='relu',
                                 name='fc_layer_4')
    fc_layer_5 = fully_connected(fc_layer_4, 2,
                                 activation='sigmoid',
                                 name='fc_layer_5')
    model = tflearn.DNN(fc_layer_5)
    model.load(model_path)
    return model


### ======================= ConvNets ===========================

def load_cnn_audio_model_buzz1(model_path):
    """
    When training on the data set BUZZ!, it had an accuracy of about 96 % and an cost of 0.26. But when validating,
    the accuracy was 43.8 %
    :param model_path: path to "cnn_audio_model_buzz1.tfl"
    :return: TFLEARN deep network model
    """
    input_layer = input_data(shape=[None, 4000, 1, 1])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=8,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    fc_layer_1 = fully_connected(pool_layer_1, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 3,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model

def load_cnn_audio_model_buzz2(model_path):
    """
    When training on the data set BUZZ2, it had an accuracy of about 99.8 % and an cost of 0.26. But when validating,
    he accuracy was 57.6 % on another data set and 37.7 % on its own data set.
    :param model_path: path to "cnn_audio_model_buzz2.tfl"
    :return: TFLEARN deep network model
    """
    input_layer = input_data(shape=[None, 4000, 1, 1])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=20,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=8,
                           filter_size=2,
                           activation='relu',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    fc_layer_1 = fully_connected(pool_layer_2, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 3,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model
    
def load_cnn_audio_model_buzz3(model_path):
    """
    When training on the data set BUZZ!, it had an accuracy of about 99.5 % and an cost of 0.13 But when validating,
    the accuracy was 54.8 % on another dataset and 81.1 % on its own dataset
    This was the best cnn audio
    :param model_path: path to "cnn_audio_model_buzz3.tfl"
    :return: TFLEARN deep network model
    """
    input_layer = input_data(shape=[None, 4000, 1, 1])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=12,
                           filter_size=2,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=10,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    fc_layer_1 = fully_connected(pool_layer_2, 256,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 3,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model

def load_cnn_image_model_bee1(model_path):
    """
    This model was trained for 40 epochs at a batch size of 8 and learning rate of 0.01
    Accuracy was 99.9 % during training and 0.0035 cost. Validation was 96.4 %.
    This is the best image cnn
    :param model_path: path to "cnn_image_model_bee1.tfl"
    :return: TFLEARN deep network model
    """
    input_layer = input_data(shape=[None, 64, 64, 3])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=8,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    fc_layer_1 = fully_connected(pool_layer_1, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model


def load_cnn_image_model_bee2_1s(model_path):
    """
    This model was trained for 35 epochs at a batch size of 8 and learning rate of 0.01
    Accuracy was 96.3 % during training and 0.0085 cost. Validation was 67.9 %
    :param model_path: path to "cnn_image_model_bee1.tfl"
    :return: TFLEARN deep network model
    """
    input_layer = input_data(shape=[None, 64, 64, 3])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=8,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=8,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    fc_layer_1 = fully_connected(pool_layer_2, 64,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model

def load_cnn_image_model_bee4(model_path):
    """
    This model was trained for 40 epochs at a batch size of 8 and learning rate of 0.01
    Accuracy was 99.6 % during training and 0.0084 cost. Validation was 71.3 %
    :param model_path: path to "cnn_image_model_bee1.tfl"
    :return: TFLEARN deep network model
    """
    input_layer = input_data(shape=[None, 64, 64, 3])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=12,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=10,
                           filter_size=3,
                           activation='sigmoid',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    conv_layer_3 = conv_2d(pool_layer_2,
                           nb_filter=8,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_3')
    pool_layer_3 = max_pool_2d(conv_layer_3, 2, name='pool_layer_3')
    fc_layer_1 = fully_connected(pool_layer_3, 64,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 128,
                                 activation='relu',
                                 name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 2,
                                 activation='softmax',
                                 name='fc_layer_3')
    model = tflearn.DNN(fc_layer_3)
    model.load(model_path)
    return model