#/usr/bin/python

from ann import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pathlib as path

##########################
# Bryson Meiling
# A01503163
# Write your code at the end of
# this file in the provided
# function stubs.
##########################

#### Libraries

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
    for i in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 1):
        print(f"Training ANN f{i}, from {lower_num_hidden_nodes} to {upper_num_hidden_nodes}")
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
    p = path.Path(r"C:\Users\Bryson M\Documents\USU\Classes\Intelligent Systems\hw04\json_nets")
    file = path.Path().joinpath(p, "net1.json")
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
    for i in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 1):
        for j in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 1):
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
    p = path.Path(r"C:\Users\Bryson M\Documents\USU\Classes\Intelligent Systems\hw04\json_nets")
    file = path.Path().joinpath(p, f"net2_{best_idx}.json")
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
    assert lower_num_hidden_nodes < upper_num_hidden_nodes
    results = {}
    anns = {}
    for i in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 1):
        for j in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 1):
            for k in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1, 1):
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
    p = path.Path(r"C:\Users\Bryson M\Documents\USU\Classes\Intelligent Systems\hw04\json_nets")
    file = path.Path().joinpath(p, f"net3_{best_idx}.json")
    anns[best_idx].save(file)
    return results
