# /usr/bin/python

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

# Libraries
json_nets_path = path.Path(r"json_nets")
train_d, valid_d, test_d = load_data_wrapper()

# auxiliary functions
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

# plotting costs and accuracies
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
    """
    This was the worst preforming network structure. Despite the size of the first layer, it couldn't compete with the
    averages of the deeper layered networks. This function iterates inside of a nested for loop that cahgnes the eta
    and lambdas to achieve the best results.
    :param lower_num_hidden_nodes: The lower bound for the hidden layer size
    :param upper_num_hidden_nodes:The upper bound for the hidden layer size
    :param cost_function: The cost function used to train
    :param num_epochs: number of times each image if passed through the network
    :param mbs: number of images from a random batch is taken
    :param eta: the momentum of training
    :param lmbda: the regularization parameter
    :param train_data: MNIST training data
    :param eval_data: MNIST evaluation data
    :return: a dictionary of the trained ANNs and their cost and accuracy scores.
    """
    assert lower_num_hidden_nodes < upper_num_hidden_nodes
    results = {}
    anns = {}
    # step was by 10 or else the training would have taken days
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

    # This finds the best network based off the best accuracy with the evaluation data. This was chosen because the
    # networks typically score less on the evaluation data there is not a risk of over fitting. Accuracy was chosen
    # over the cost function as the determining factor because accuracy is better for a real world application where the
    # user wants the networks to correctly define the handwritten number
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
    """
    In general, increasing the number of nodes in the hidden layers increased the performance. This was not always the
    case, but as a general rule, it held true. Training was slower than expected, but it did not take as long at the
    three layer networks. The for loops changed step by 10 each time to reduce the number of permutations from
    astronomical numbers to something more manageable in a week long time frame.
    :param lower_num_hidden_nodes: The lower bound for the hidden layer size
    :param upper_num_hidden_nodes:The upper bound for the hidden layer size
    :param cost_function: The cost function used to train
    :param num_epochs: number of times each image if passed through the network
    :param mbs: number of images from a random batch is taken
    :param eta: the momentum of training
    :param lmbda: the regularization parameter
    :param train_data: MNIST training data
    :param eval_data: MNIST evaluation data
    :return: a dictionary of the trained ANNs and their cost and accuracy scores.
    """
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

    # This finds the best network based off the best accuracy with the evaluation data. This was chosen because the
    # networks typically score less on the evaluation data there is not a risk of over fitting. Accuracy was chosen
    # over the cost function as the determining factor because accuracy is better for a real world application where the
    # user wants the networks to correctly define the handwritten number
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
    loop is 10, which reduces the permutations to 512. Even from the beginning of the testing, a deeper three layer
    network outpreforms the 2 or 1 layer networks. Average scores were in the 96%s where as the shallower networks
    were in the 94 - 95% region.
    :param lower_num_hidden_nodes: The lower bound for the hidden layer size
    :param upper_num_hidden_nodes:The upper bound for the hidden layer size
    :param cost_function: The cost function used to train
    :param num_epochs: number of times each image if passed through the network
    :param mbs: number of images from a random batch is taken
    :param eta: the momentum of training
    :param lmbda: the regularization parameter
    :param train_data: MNIST training data
    :param eval_data: MNIST evaluation data
    :return: a dictionary of the trained ANNs and their cost and accuracy scores.
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

    # This finds the best network based off the best accuracy with the evaluation data. This was chosen because the
    # networks typically score less on the evaluation data there is not a risk of over fitting. Accuracy was chosen
    # over the cost function as the determining factor because accuracy is better for a real world application where the
    # user wants the networks to correctly define the handwritten number
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

for e in etas:
    for lam in lambdas:
        print(f"eta = {e},  lambda = {lam}")
        collect_2_hidden_layer_net_stats(30, 100, CrossEntropyCost, 30, 10, e, lam, train_d, test_d)
