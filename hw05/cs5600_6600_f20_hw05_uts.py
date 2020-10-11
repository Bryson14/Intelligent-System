#/usr/bin/python

###########################################
# Unit Tests for Assignment 5
# bugs to vladimir kulyukin via canvas
###########################################

import json
import random
import sys
import numpy as np
import unittest
import tensorflow as tf
import tflearn
#from tflearn.layers.core import input_data, fully_connected
#from tflearn.layers.conv import conv_2d, max_pool_2d
#from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist
from cs5600_6600_f20_hw05 import *

### Let's load MNIST and reshape train, test, and validation sets.
X, Y, testX, testY = mnist.load_data(one_hot=True)
testX, testY = tflearn.data_utils.shuffle(testX, testY)
trainX, trainY = X[0:50000], Y[0:50000]
validX, validY = X[50000:], Y[50000:]
validX, validY = tflearn.data_utils.shuffle(validX, validY)
trainX = trainX.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])
validX = validX.reshape([-1, 28, 28, 1])

## change this directory accordingly.

my_path = Path.joinpath(Path.cwd(), 'nets\\')
assert my_path.is_dir()
NET_PATH = str(my_path.absolute()) + "\\"

class cs5600_6600_f20_hw05_uts(unittest.TestCase):

    def test_ut01(self):
        tf.reset_default_graph()
        model_slide22 = make_tfl_convnet_slide22()
        model_name = 'ConvNet_Slide22.tfl'
        assert model_slide22 is not None
        fit_tfl_model(model_slide22,
                      trainX, trainY, testX, testY,
                      model_name, NET_PATH,
                      n_epoch=5, mbs=10)

    def test_ut02(self):
        tf.reset_default_graph()
        model_slide22 = load_tfl_convnet_slide22(NET_PATH + 'ConvNet_Slide22.tfl')
        assert model_slide22 is not None
        i = np.random.randint(0, len(validX)-1)
        prediction = model_slide22.predict(validX[i].reshape([-1, 28, 28, 1]))
        print('raw prediction   = {}'.format(prediction))
        print('raw ground truth = {}'.format(validY[i]))
        prediction   = np.argmax(prediction, axis=1)[0]
        ground_truth = np.argmax(validY[i])
        print('ground truth = {}'.format(ground_truth))
        print('prediction   = {}'.format(prediction))
        print(prediction == ground_truth)
    
    def test_ut03(self):
        tf.reset_default_graph()        
        model_slide22 = load_tfl_convnet_slide22(NET_PATH + 'ConvNet_Slide22.tfl')
        assert model_slide22 is not None
        acc = test_tfl_model(model_slide22, validX, validY)
        print('ConvNet Slide 22 Acc = {}'.format(acc))
        
    def test_ut04(self):
        tf.reset_default_graph()
        deeper_model = make_deeper_tfl_convnet()
        model_name = 'Deeper_ConvNet.tfl'
        assert deeper_model is not None
        fit_tfl_model(deeper_model,
                      trainX, trainY, testX, testY,
                      model_name, NET_PATH,
                      n_epoch=5, mbs=10)

    def test_ut05(self):
        tf.reset_default_graph()
        model_name = 'Deeper_ConvNet.tfl'        
        deeper_model = load_deeper_tfl_convnet(NET_PATH + model_name)
        assert deeper_model is not None
        acc = test_tfl_model(deeper_model, validX, validY)
        print('Deeper ConvNet Acc = {}'.format(acc))

    def test_ut06(self):
        tf.reset_default_graph()
        shallow_model = make_shallow_tfl_ann()
        model_name = 'Shallow_ANN.tfl'
        assert shallow_model is not None
        fit_tfl_model(shallow_model,
                      trainX, trainY, testX, testY,
                      model_name, NET_PATH,
                      n_epoch=10, mbs=10)

    def test_ut07(self):
        tf.reset_default_graph()
        model_name = 'Shallow_ANN.tfl'        
        shallow_model = load_shallow_tfl_ann(NET_PATH + model_name)
        assert shallow_model is not None
        acc = test_tfl_model(shallow_model, validX, validY)
        print('Shallow ConvNet Acc = {}'.format(acc))


if __name__ == '__main__':
    unittest.main()
    pass
