#!/usr/bin/python

"""
1. The performance report and confusion matrix of your top performing DT;
2. The performance report and confusion matrix of your top performing RF. Be sure to specify
the number of DTs in your RF;
3. Your brief answer to the question: Do you think that ensemble learning makes a dierence in
going from single DTs to RFs?
4. Your brief answers to these two questions. How does the accuracy of your best MNIST ConvNet
compare to the accuracies of your best MNIST DT and RF (i.e., their average weighted F1
scores)? Have you drawn any conclusions from this comparison?

"""
####################################################
# module: mnist_digits_random_forest.py
# description: Testing random forest for MNIST
# bugs to vladimir dot kulyukin via canvas
####################################################

from sklearn import tree, metrics
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from mnist_loader import load_data_wrapper

mnist_train_data, mnist_test_data, mnist_valid_data = \
                  load_data_wrapper()

mnist_train_data_dc = np.zeros((50000, 784))
mnist_test_data_dc  = np.zeros((10000, 784))
mnist_valid_data_dc = np.zeros((10000, 784))

mnist_train_target_dc = None
mnist_test_target_dc  = None
mnist_valid_target_dc = None

def reshape_mnist_aux(mnist_data, mnist_data_dc):
    '''auxiliary function to reshape MNIST data for sklearn.'''
    for i in range(len(mnist_data)):
        mnist_data_dc[i] = mnist_data[i][0].reshape((784,))

def reshape_mnist_data():
    '''reshape all MNIST data for sklearn.'''
    global mnist_train_data
    global mnist_train_data_dc
    global mnist_test_data
    global mnist_test_data_dc
    global mnist_valid_data
    global mnist_valid_data_dc
    reshape_mnist_aux(mnist_train_data, mnist_train_data_dc)
    reshape_mnist_aux(mnist_test_data,  mnist_test_data_dc)
    reshape_mnist_aux(mnist_valid_data, mnist_valid_data_dc)

def reshape_mnist_target(mnist_data):
    '''reshape MNIST target given data.'''
    return np.array([np.argmax(mnist_data[i][1])
                    for i in range(len(mnist_data))])

def reshape_mnist_target2(mnist_data):
    '''another function for reshaping MNIST target given data.'''
    return np.array([mnist_data[i][1] for i in range(len(mnist_data))])

def prepare_mnist_data():
    '''reshape and prepare MNIST data for sklearn.'''
    global mnist_train_data
    global mnist_test_data
    global mnist_valid_data
    reshape_mnist_data()

    ### make sure that train, test, and valid data are reshaped
    ### correctly.
    for i in range(len(mnist_train_data)):
        assert np.array_equal(mnist_train_data[i][0].reshape((784,)),
                              mnist_train_data_dc[i])

    for i in range(len(mnist_test_data)):
        assert np.array_equal(mnist_test_data[i][0].reshape((784,)),
                              mnist_test_data_dc[i])

    for i in range(len(mnist_valid_data)):
        assert np.array_equal(mnist_valid_data[i][0].reshape((784,)),
                              mnist_valid_data_dc[i])

def prepare_mnist_targets():
    '''reshape and prepare MNIST targets for sklearn.'''
    global mnist_train_target_dc
    global mnist_test_target_dc
    global mnist_valid_target_dc    
    mnist_train_target_dc = reshape_mnist_target(mnist_train_data)
    mnist_test_target_dc  = reshape_mnist_target2(mnist_test_data)
    mnist_valid_target_dc = reshape_mnist_target2(mnist_valid_data) 

def fit_validate_dt():

    clf = tree.DecisionTreeClassifier(random_state=0)
    dtr = clf.fit(mnist_train_data_dc, mnist_train_target_dc)

    valid_preds = dtr.predict(mnist_valid_data_dc)
    print(metrics.classification_report(mnist_valid_target_dc, valid_preds))

    cm1 = confusion_matrix(mnist_valid_target_dc, valid_preds)

    test_preds = dtr.predict(mnist_test_data_dc)
    print(metrics.classification_report(mnist_test_target_dc, test_preds))

    cm2 = confusion_matrix(mnist_test_target_dc, test_preds)

def fit_validate_dts(num_dts):
    # your code here
    pass  

def fit_validate_rf(num_dts):
    # your code here
    pass

def fit_validate_rfs(low_nt, high_nt):
    # your code here
    pass

## Let's prepare MNIST data for unit tests.        
prepare_mnist_data()
prepare_mnist_targets()

'''
if __name__ == '__main__':
    prepare_mnist_data()
    prepare_mnist_targets()
'''
    


