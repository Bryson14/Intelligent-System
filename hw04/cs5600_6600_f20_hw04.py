#/usr/bin/python

from ann import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pathlib as path
from mnist_loader import load_data_wrapper

##########################
# Bryson Meiling
# A01503163
# Write your code at the end of
# this file in the provided
# function stubs.
##########################

#### Libraries
json_nets_path = path.Path(r"json_nets")
train_d, valid_d, test_d = load_data_wrapper()

#### auxiliary functions
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of ann.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = ann(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### plotting costs and accuracies
def plot_costs(eval_costs, train_costs, num_epochs):
    plt.title('Evaluation Cost (EC) and Training Cost (TC)')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    epochs = [i for i in range(num_epochs)]
    plt.plot(epochs, eval_costs, label='EC', c='g')
    plt.plot(epochs, train_costs, label='TC', c='b')
    plt.grid()    
    plt.autoscale(tight=True)
    plt.legend(loc='best')
    plt.show()

def plot_accuracies(eval_accs, train_accs, num_epochs):
    plt.title('Evaluation Acc (EA) and Training Acc (TC)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    epochs = [i for i in range(num_epochs)]
    plt.plot(epochs, eval_accs, label='EA', c='g')
    plt.plot(epochs, train_accs, label='TA', c='b')
    plt.grid()
    plt.autoscale(tight=True)
    plt.legend(loc='best')    
    plt.show()

def collect_1_hidden_layer_net_stats(lower_num_hidden_nodes,
                                     upper_num_hidden_nodes,
                                     cost_function,
                                     num_epochs,
                                     mbs,
                                     eta,
                                     lmbda,
                                     train_data,
                                     eval_data):
    assert lower_num_hidden_nodes < upper_num_hidden_nodes
    results = {}
    anns = {}
    for i in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 10):
        print(f"Training ANN {i}, from {lower_num_hidden_nodes} to {upper_num_hidden_nodes}")
        net = ann([784, i, 10], cost=cost_function)
        net_stats = net.mini_batch_sgd(train_data,
                                       num_epochs, mbs, eta, lmbda,
                                       evaluation_data=eval_data,
                                       monitor_evaluation_cost=True,
                                       monitor_evaluation_accuracy=True,
                                       monitor_training_cost=True,
                                       monitor_training_accuracy=True)
        results[i] = net_stats
        anns[i] = net

    best_accuracy = 0.0
    best_idx = 0
    for key in results:
        if results[key][3][-1] > best_accuracy:
            best_accuracy = results[key][3][-1]
            best_idx = key
    file = path.Path().joinpath(json_nets_path, f"net1_{best_idx}_eta-{eta}_lambda-{lmbda}.json")
    anns[best_idx].save(file)
    return results


def collect_2_hidden_layer_net_stats(lower_num_hidden_nodes,
                                     upper_num_hidden_nodes,
                                     cost_function,
                                     num_epochs,
                                     mbs,
                                     eta,
                                     lmbda,
                                     train_data,
                                     eval_data):
    assert lower_num_hidden_nodes < upper_num_hidden_nodes
    results = {}
    anns = {}
    for i in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 10):
        for j in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 10):
            print(f"Training ANN {i}_{j}, from {lower_num_hidden_nodes} to {upper_num_hidden_nodes}")
            net = ann([784, i, j, 10], cost=cost_function)
            net_stats = net.mini_batch_sgd(train_data,
                                           num_epochs, mbs, eta, lmbda,
                                           evaluation_data=eval_data,
                                           monitor_evaluation_cost=True,
                                           monitor_evaluation_accuracy=True,
                                           monitor_training_cost=True,
                                           monitor_training_accuracy=True)
            results[f"{i}_{j}"] = net_stats
            anns[f"{i}_{j}"] = net

    best_accuracy = 0.0
    best_idx = ""
    for key in results:
        if results[key][3][-1] > best_accuracy:
            best_accuracy = results[key][3][-1]
            best_idx = key
    file = path.Path().joinpath(json_nets_path, f"net2_{best_idx}_eta-{eta}_lambda-{lmbda}.json")
    anns[best_idx].save(file)
    return results
    

def collect_3_hidden_layer_net_stats(lower_num_hidden_nodes,
                                     upper_num_hidden_nodes,
                                     cost_function,
                                     num_epochs,
                                     mbs,
                                     eta,
                                     lmbda,
                                     train_data,
                                     eval_data):
    """
    For the assignment, if there was to be a triple for loop between 30 to 100 by 1, that would be 343,000 permutations.
    This is an insane amount and is not feasible for a week long assignment. Therefore, the step size of each for
    loop is 10, which reduces the permutations to 512.
    :param lower_num_hidden_nodes:
    :param upper_num_hidden_nodes:
    :param cost_function:
    :param num_epochs:
    :param mbs:
    :param eta:
    :param lmbda:
    :param train_data:
    :param eval_data:
    :return:
    """
    assert lower_num_hidden_nodes < upper_num_hidden_nodes
    results = {}
    anns = {}
    for i in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 10):
        for j in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 10):
            for k in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 10):
                print(f"Training ANN {i}_{j}_{k}, from {lower_num_hidden_nodes} to {upper_num_hidden_nodes}")
                net = ann([784, i, j, k, 10], cost=cost_function)
                net_stats = net.mini_batch_sgd(train_data,
                                               num_epochs, mbs, eta, lmbda,
                                               evaluation_data=eval_data,
                                               monitor_evaluation_cost=True,
                                               monitor_evaluation_accuracy=True,
                                               monitor_training_cost=True,
                                               monitor_training_accuracy=True)
                results[f"{i}_{j}_{k}"] = net_stats
                anns[f"{i}_{j}_{k}"] = net

    best_accuracy = 0.0
    best_idx = ""
    for key in results:
        if results[key][3][-1] > best_accuracy:
            best_accuracy = results[key][3][-1]
            best_idx = key
    file = path.Path().joinpath(json_nets_path, f"net3_{best_idx}_eta-{eta}_lambda-{lmbda}.json")
    anns[best_idx].save(file)
    return results


etas = {0.1, 0.3, 0.5}
lambdas = {0.01, 0.1, 1}

for eta in etas:
    for lam in lambdas:
        print(f"eta = {eta},  lambda = {lam}")
        collect_2_hidden_layer_net_stats(30, 100, CrossEntropyCost, 30, 10, eta, lam, train_d, test_d)