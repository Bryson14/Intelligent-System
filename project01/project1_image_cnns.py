########################################################
# module: project1_image_cnns.py
# authors: vladimir kulyukin, nikhil ganta
# descrption: starter code for image CNNs for project 1
# bugs to vladimir and nikhil in canvas
########################################################

import pickle
import pathlib
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

## we need this to load the pickled data into Python.
def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

## Paths to all datasets. Change accordingly.
dataset_path = pathlib.Path("datasets")
BEE1_path = str(dataset_path.joinpath('BEE1').absolute())
BEE2_1S_path = str(dataset_path.joinpath('BEE2_1S').absolute())
BEE4_path =  str(dataset_path.joinpath('BEE4').absolute())

## let's load BEE1
base_path = BEE1_path + "\\"
print('loading datasets from {}...'.format(base_path))
BEE1_train_X = load(base_path + 'train_X.pck')
BEE1_train_Y = load(base_path + 'train_Y.pck')
BEE1_test_X = load(base_path + 'test_X.pck')
BEE1_test_Y = load(base_path + 'test_Y.pck')
BEE1_valid_X = load(base_path + 'valid_X.pck')
BEE1_valid_Y = load(base_path + 'valid_Y.pck')
print(BEE1_train_X.shape)
print(BEE1_train_Y.shape)
print(BEE1_test_X.shape)
print(BEE1_test_Y.shape)
print(BEE1_valid_X.shape)
print(BEE1_valid_Y.shape)
print('datasets from {} loaded...'.format(base_path))
BEE1_train_X = BEE1_train_X.reshape([-1, 64, 64, 3])
BEE1_test_X = BEE1_test_X.reshape([-1, 64, 64, 3])

## to make sure that the dimensions of the
## examples and targets are the same.
assert BEE1_train_X.shape[0] == BEE1_train_Y.shape[0]
assert BEE1_test_X.shape[0]  == BEE1_test_Y.shape[0]
assert BEE1_valid_X.shape[0] == BEE1_valid_Y.shape[0]

## let's load BEE2_1S
base_path = BEE2_1S_path + "\\"
print('loading datasets from {}...'.format(base_path))
BEE2_1S_train_X = load(base_path + 'train_X.pck')
BEE2_1S_train_Y = load(base_path + 'train_Y.pck')
BEE2_1S_test_X = load(base_path + 'test_X.pck')
BEE2_1S_test_Y = load(base_path + 'test_Y.pck')
BEE2_1S_valid_X = load(base_path + 'valid_X.pck')
BEE2_1S_valid_Y = load(base_path + 'valid_Y.pck')
print(BEE2_1S_train_X.shape)
print(BEE2_1S_train_Y.shape)
print(BEE2_1S_test_X.shape)
print(BEE2_1S_test_Y.shape)
print(BEE2_1S_valid_X.shape)
print(BEE2_1S_valid_Y.shape)
print('datasets from {} loaded...'.format(base_path))
BEE2_1S_train_X = BEE2_1S_train_X.reshape([-1, 64, 64, 3])
BEE2_1S_test_X = BEE2_1S_test_X.reshape([-1, 64, 64, 3])

assert BEE2_1S_train_X.shape[0] == BEE2_1S_train_Y.shape[0]
assert BEE2_1S_test_X.shape[0]  == BEE2_1S_test_Y.shape[0]
assert BEE2_1S_valid_X.shape[0] == BEE2_1S_valid_Y.shape[0]

## let's load BEE4
base_path = BEE4_path + "\\"
print('loading datasets from {}...'.format(base_path))
BEE4_train_X = load(base_path + 'train_X.pck')
BEE4_train_Y = load(base_path + 'train_Y.pck')
BEE4_test_X = load(base_path + 'test_X.pck')
BEE4_test_Y = load(base_path + 'test_Y.pck')
BEE4_valid_X = load(base_path + 'valid_X.pck')
BEE4_valid_Y = load(base_path + 'valid_Y.pck')
print(BEE4_train_X.shape)
print(BEE4_train_Y.shape)
print(BEE4_test_X.shape)
print(BEE4_test_Y.shape)
print(BEE4_valid_X.shape)
print(BEE4_valid_Y.shape)
print('datasets from {} loaded...'.format(base_path))
BEE4_train_X = BEE4_train_X.reshape([-1, 64, 64, 3])
BEE4_test_X = BEE4_test_X.reshape([-1, 64, 64, 3])

assert BEE4_train_X.shape[0] == BEE4_train_Y.shape[0]
assert BEE4_test_X.shape[0]  == BEE4_test_Y.shape[0]
assert BEE4_valid_X.shape[0] == BEE4_valid_Y.shape[0]

### here's an example of how to make an CNN with tflearn.
def make_image_cnn_model():
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
    network = regression(fc_layer_3, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.1)
    model = tflearn.DNN(network)
    return model

def load_image_cnn_model(model_path):
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

def test_tfl_image_cnn_model(network_model, validX, validY):
    results = []
    for i in range(len(validX)):
        prediction = network_model.predict(validX[i].reshape([-1, 64, 64, 3]))
        results.append(np.argmax(prediction, axis=1)[0] == \
                       np.argmax(validY[i]))
    return float(sum((np.array(results) == True))) / float(len(results))

###  train a tfl cnn model on train_X, train_Y, test_X, test_Y.
def train_tfl_image_cnn_model(model, train_X, train_Y, test_X, test_Y, num_epochs=2, batch_size=10):
  tf.reset_default_graph()
  model.fit(train_X, train_Y, n_epoch=num_epochs,
            shuffle=True,
            validation_set=(test_X, test_Y),
            show_metric=True,
            batch_size=batch_size,
            run_id='image_cnn_model')

### validating is testing on valid_X and valid_Y.
def validate_tfl_image_cnn_model(model, valid_X, valid_Y):
    return test_tfl_image_cnn_model(model, valid_X, valid_Y)
  
bee1_net_name = "cnn_image_model_bee1.tfl"
bee2_net_name = "cnn_image_model_bee2_1s.tfl"
bee3_net_name = "cnn_image_model_bee4.tfl"

# # making and training
# model = make_image_cnn_model()
# train_tfl_image_cnn_model(model, BEE4_train_X, BEE4_train_Y, BEE4_test_X, BEE4_test_Y, 40, 8)
# model.save("nets\\cnn_image_model_bee4.tfl")

# validation
from load_nets import load_cnn_image_model_bee1, load_cnn_image_model_bee2_1s, load_cnn_image_model_bee4
model = load_cnn_image_model_bee4("nets\\cnn_image_model_bee4.tfl")
print(validate_tfl_image_cnn_model(model, BEE4_valid_X, BEE4_valid_Y))
