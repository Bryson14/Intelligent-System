#!/usr/bin/python

"""
1. The performance report and confusion matrix of your top performing DT;


                  precision    recall  f1-score   support

               0       0.92      0.94      0.93       980
               1       0.94      0.96      0.95      1135
               2       0.89      0.84      0.86      1032
               3       0.83      0.85      0.84      1010
               4       0.87      0.87      0.87       982
               5       0.81      0.82      0.82       892
               6       0.89      0.89      0.89       958
               7       0.89      0.91      0.90      1028
               8       0.82      0.79      0.81       974
               9       0.83      0.85      0.84      1009

        accuracy                           0.87     10000
       macro avg       0.87      0.87      0.87     10000
    weighted avg       0.87      0.87      0.87     10000

                  precision    recall  f1-score   support

               0       0.93      0.92      0.93       991
               1       0.94      0.95      0.95      1064
               2       0.87      0.87      0.87       990
               3       0.85      0.86      0.85      1030
               4       0.89      0.87      0.88       983
               5       0.85      0.82      0.83       915
               6       0.91      0.92      0.92       967
               7       0.89      0.92      0.91      1090
               8       0.84      0.81      0.83      1009
               9       0.82      0.85      0.83       961

        accuracy                           0.88     10000
       macro avg       0.88      0.88      0.88     10000
    weighted avg       0.88      0.88      0.88     10000

    [[ 916    2   13   11    4   10   12    3   13    7]
     [   1 1015    8    5    1    3    4   11   13    3]
     [  11   14  857   21    8    6   15   27   21   10]
     [   5   10   21  881   10   36    7   16   19   25]
     [   6    4    7    4  859    8    8   17   24   46]
     [  16    7   10   47   13  749   19    5   25   24]
     [   7    4   12    4   13   17  890    2   15    3]
     [   1    6   19   10    7    7    1 1005    6   28]
     [  15   13   23   31   18   30   12   10  822   35]
     [   7    6   12   17   34   16    5   27   20  817]]

2. The performance report and confusion matrix of your top performing RF. Be sure to specify
the number of DTs in your RF;
    There were 491 decision trees in the random forest.


                  precision    recall  f1-score   support

               0       0.97      0.99      0.98       980
               1       0.99      0.99      0.99      1135
               2       0.96      0.97      0.96      1032
               3       0.96      0.96      0.96      1010
               4       0.98      0.97      0.97       982
               5       0.98      0.96      0.97       892
               6       0.98      0.98      0.98       958
               7       0.97      0.96      0.97      1028
               8       0.96      0.96      0.96       974
               9       0.96      0.95      0.95      1009

        accuracy                           0.97     10000
       macro avg       0.97      0.97      0.97     10000
    weighted avg       0.97      0.97      0.97     10000

                  precision    recall  f1-score   support

               0       0.98      0.99      0.99       991
               1       0.98      0.99      0.99      1064
               2       0.97      0.98      0.97       990
               3       0.97      0.97      0.97      1030
               4       0.98      0.98      0.98       983
               5       0.97      0.95      0.96       915
               6       0.98      0.99      0.99       967
               7       0.98      0.98      0.98      1090
               8       0.96      0.96      0.96      1009
               9       0.96      0.95      0.96       961

        accuracy                           0.97     10000
       macro avg       0.97      0.97      0.97     10000
    weighted avg       0.97      0.97      0.97     10000

    [[ 980    0    3    0    0    0    2    0    4    2]
     [   0 1052    5    1    2    1    1    0    2    0]
     [   2    1  969    0    2    1    1    6    6    2]
     [   1    0    4 1000    0    8    0    3   11    3]
     [   0    4    0    0  959    0    2    1    1   16]
     [   4    0    4   14    2  871   12    1    5    2]
     [   1    0    0    0    1    3  960    0    2    0]
     [   1    5    5    1    3    0    0 1065    0   10]
     [   1    7    6    5    0    6    4    1  973    6]
     [   5    2    2    8    8    4    0    8    8  916]]

3. Your brief answer to the question: Do you think that ensemble learning makes a dierence in
going from single DTs to RFs?
    Yes this makes a huge difference because its highly improbable that one decision tree will be able to be trained
    to classify all the possible combinations that are in a data set. Using the strong law of large numbers,
    we can be sure that our answers converge at a top accuracy for the the decision tree's design.

4. Your brief answers to these two questions. How does the accuracy of your best MNIST ConvNet
compare to the accuracies of your best MNIST DT and RF (i.e., their average weighted F1
scores)? Have you drawn any conclusions from this comparison?
    The best conv net from hw 5 had an accuracy of 98.24 % whereas the random forest had a accuracy of 98 %. These are
    essentially the same. This shows to me that in many instances of classsifying, random forests are able to train
    quicker and be just as accurate as conv nets that have be trained for much more time. Classification might be
    a task better suited for random forest.

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
    print(cm2)

def fit_validate_dts(num_dts):
    for i in range(num_dts):
        print(f"++++++++ DT # {i + 1} of {num_dts} +++++++++++")
        fit_validate_dt()


def fit_validate_rf(num_dts):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=num_dts,
                                 random_state=random.randint(0, 1000))
    rf = clf.fit(mnist_train_data_dc, mnist_train_target_dc)
    valid_preds = rf.predict(mnist_valid_data_dc)

    print(metrics.classification_report(mnist_valid_target_dc, valid_preds))

    cm1 = confusion_matrix(mnist_valid_target_dc, valid_preds)

    test_preds = rf.predict(mnist_test_data_dc)
    print(metrics.classification_report(mnist_test_target_dc, test_preds))

    cm2 = confusion_matrix(mnist_test_target_dc, test_preds)
    print(cm2)

def fit_validate_rfs(low_nt, high_nt):
    for i in range(low_nt, high_nt, 10):
        print(f"++++++++ RF # {i + 1} of ({low_nt}-{high_nt}) +++++++++++")
        fit_validate_rf(i)

## Let's prepare MNIST data for unit tests.        
prepare_mnist_data()
prepare_mnist_targets()
# fit_validate_rfs(10, 500)

'''
if __name__ == '__main__':
    prepare_mnist_data()
    prepare_mnist_targets()
'''
    


