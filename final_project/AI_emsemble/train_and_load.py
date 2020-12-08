import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from polygon_data import polygon_wrapper
import pandas as pd
from datetime import datetime, timedelta


def label_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the data from the api and adds a new column, action. This looks at the yesterday and if the closing price is
    higher than today's closing price, it puts a '-1' (sell) in the action column. If yesterday's closing price is
    lower that today's closing price, it puts a '1' (buy) in the action column. This labels the data and allows for
    reinforcement learning for the nets and forests.
    :param data: A pandas dataframe with h (high), l (low), o (open), c (close), t (datetime), v (volume) columns.
    Assumes that the column t is sorted in either ascending or descending order
    :return: A pandas data frame with two columns, a 'id' and an 'action' column. The 'id' will match the respective
    id row of the input dataframe.
    """

    descending = False
    test = data[:2]["c"].to_list()
    test2 = data[:2].sort_values("t")["c"].to_list()
    if test == test2:
        descending = True
    else:
        # the data is in ascending order
        data = data[::-1]

    actions = np.zeros((data.axes[0].size,), dtype=int)
    index = 0
    previous = 0.0
    for row in data['c']:
        # Not taking into account the first day that has no yesterday reference (0 by default)
        if previous != 0.0:
            # if today is higher than yesterday, the system should have bought (1)
            if row >= previous:
                actions[index] = 1
            # if today is lower than yesterday, the system should have sold (0)
            else:
                pass
        previous = row
        index += 1

    data['action'] = actions

    if not descending:
        # flip the data frame back into the ascending order it came
        return data[::-1]

    return data


def make_cnn_model():
    """
    Defines the CNN model structure and returns the model
    :return: TFLearn Deep Neural Network
    """
    # inputs are v,vw,o,c,h,l,t,n (X) label is action (Y)
    input_layer = input_data(shape=[None, 7])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=12,
                           filter_size=2,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=10,
                           filter_size=2,
                           activation='sigmoid',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    fc_layer_1 = fully_connected(pool_layer_2, 64,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_1 = tflearn.dropout(fc_layer_1, 0.5)
    fc_layer_2 = fully_connected(fc_layer_1, 64,
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

def load_cnn_model(model_path):
    input_layer = input_data(shape=[None, 7])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=12,
                           filter_size=2,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=10,
                           filter_size=2,
                           activation='sigmoid',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    fc_layer_1 = fully_connected(pool_layer_2, 64,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_1 = tflearn.dropout(fc_layer_1, 0.5)
    fc_layer_2 = fully_connected(fc_layer_1, 64,
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

def other_network():
    network = input_data(shape=[None, 7], name='Input_layer')
    network = fully_connected(network, 5, activation='relu', name='Hidden_layer_1')
    network = fully_connected(network, 1, activation='linear', name='Output_layer')
    network = regression(network, batch_size=64, optimizer='sgd', learning_rate=0.2, loss='mean_square', metric='R2')
    model = tflearn.DNN(network)
    return model

def test_tfl_cnn_model(network_model, valid_X, valid_Y):
    results = []
    for i in range(len(valid_X)):
        prediction = network_model.predict(valid_X[i])
        results.append(np.argmax(prediction, axis=1)[0] == \
                       np.argmax(valid_Y[i]))
    return float(sum((np.array(results) == True))) / float(len(results))


#  train a tfl model on train_X, train_Y, test_X, test_Y.
def train_tfl_cnn_model(model, train_X, train_Y, test_X, test_Y, num_epochs=2, batch_size=10):
  tf.reset_default_graph()
  model.fit(train_X, train_Y, n_epoch=num_epochs,
            shuffle=True,
            validation_set=(test_X, test_Y),
            show_metric=True,
            batch_size=batch_size,
            run_id='spy_cnn_model')

# validating is testing on valid_X and valid_Y.
def validate_tfl_audio_cnn_model(model, validX, validY):
    return test_tfl_cnn_model(model, validX, validY)


# reading data
df = pd.read_csv("spy_minute_2015-01-01_2020-10-30.csv", delimiter=",")
df.pop("Unnamed: 0")  # the csv is loaded with the indexes accidentally made into another column
df.pop('t')  # time in not relevant when training
num_rows = df.shape[0]

# splitting it into test (19 K) , valid (100 K), and training data (900 K)
size_train = int(num_rows * 0.8)
size_valid = int(num_rows * 0.1)
size_test = num_rows - size_train - size_valid

trainX = df[df.index < size_train]
testX = df[np.logical_and(size_train <= df.index, df.index < (size_train + size_test))]
validX = df[df.index >= (size_train + size_test)]

# popping of the labels Y

trainY = trainX.pop("action")
testY = testX.pop("action")
validY = validX.pop("action")

# turning into numpy arrays for tflearn to handle
trainX = trainX.to_numpy()
trainY = trainY.to_numpy()
testX = testX.to_numpy()
testY = testY.to_numpy()
validX = validX.to_numpy()
validY = validY.to_numpy()

# reshaping for training
print('shape of x, y ->', testX.shape, testY.shape)


network = other_network()
train_tfl_cnn_model(network, trainX, trainY, testX, testY)
network.save("nets//cnn_spy_net.tfl")
