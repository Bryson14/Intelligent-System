#/usr/bin/python

############################
# Module: cs5600_6600_f20_hw03.py
# Your name
# Your A#
############################

# from network import Network
from mnist_loader import load_data_wrapper
from ann import ann
import random
import pickle as cPickle
import pathlib
import numpy as np

# load training, validation, and testing MNIST data
train_d, valid_d, test_d = load_data_wrapper()
input_layer = 784
output_layer = 10

def train_1_hidden_layer_anns(lwr=10, upr=50, eta=0.25, mini_batch_size=10, num_epochs=10):
    assert 100 >= upr > lwr >= 10
    assert lwr % 10 == 0 and upr % 10 == 0  # divisible by 10
    while lwr <= upr:
        print(f"==== Training {input_layer}x{lwr}x{output_layer} ANN ======")
        net = ann([input_layer, lwr, output_layer])
        net.mini_batch_sgd(train_d, num_epochs, mini_batch_size, eta, test_d)

        print(f"==== Training {input_layer}x{lwr}x{output_layer} ANN - DONE - eta:{eta} ======")
        lwr += 10

def train_2_hidden_layer_anns(lwr=10, upr=50, eta=0.25, mini_batch_size=10, num_epochs=10):
    assert 100 >= upr > lwr >= 10
    assert lwr % 10 == 0 and upr % 10 == 0  # divisible by 10
    hl1 = 10

    eta_vals = (2, 1.5, 1.0, 0.5, 0.25)
    results = np.zeros([6,6], dtype=float)
    results[0,0] = -1
    results[1:,0] = np.array(list(eta_vals))
    for y in range(len(results) - 1):
        results[0,y + 1] = 10 * (y + 1)
    for i in range(len(eta_vals)):
        eta = eta_vals[i]
        hl2 = lwr
        while hl2 <= upr:
            print(f"==== Training {input_layer}x{hl1}x{hl2}x{output_layer} - eta:{eta} ANN ======")
            net = ann([input_layer, hl1, hl2, output_layer])
            net.mini_batch_sgd(train_d, num_epochs, mini_batch_size, eta, test_d)
            result = net.evaluate(test_d)
            results[i+1, hl2//10] = result
            print(f"==== Training {input_layer}x{hl1}x{hl2}x{output_layer} ANN - DONE - eta:{eta} ======")
            hl2 += 10

    print(f"Training DONE {input_layer}x{hl1}x10-50x{output_layer}")
    print(results)


# define your networks
net1 = ann([input_layer, 50, 60, output_layer])     # 4 layers
net2 = ann([input_layer, 50, 60, 30, output_layer])         # 5 layers
net3 = ann([input_layer, 10, 30, output_layer])             # 3 layers
net4 = ann([input_layer, 60, 30, 20, 40, 20, output_layer])     # 7 layers
net5 = ann([input_layer, 10, 50, 60, 30, output_layer])     # 6 layers

# define an ensemble of 5 nets
networks = (net1, net2, net3, net4, net5)
eta_vals = (0.1, 0.25, 0.3, 0.4, 0.5)

mini_batch_sizes = (5, 10, 15, 20)

# train networks
def train_nets(networks, eta_vals, mini_batch_sizes, num_epochs, path):
    assert pathlib.Path(path).is_dir()
    eta = np.random.choice(eta_vals)
    mini_batch_size = np.random.choice(mini_batch_sizes)

    for network in networks:
        net_name = "net_"
        for size in network.sizes:
            net_name += str(size) + "x"
        net_name = net_name[:-1] + f"_mbs-{mini_batch_size}_eta-{int(eta*100)}.pkl"

        network.mini_batch_sgd(train_d, num_epochs,mini_batch_size, eta, test_d)
        print(f"Trained Network {net_name}")

        p = pathlib.Path(path).absolute()
        p = pathlib.Path.joinpath(p, net_name)
        with open(p, 'wb+') as file:
            cPickle.dump(network, file)

def load_nets(path):
    p = pathlib.Path(path)
    assert p.is_dir()
    files = []
    for x in p.iterdir():
        if x.is_file() and str(x).endswith("pkl"):
            files.append(x)
        else:
            print(f"error, dir found in {p}")
            exit()

    loaded_network = [(file.name, cPickle.load(open(file, "rb"))) for file in files]
    return loaded_network

# evaluate net ensemble.
def evaluate_net_ensemble(net_ensemble, test_data):
    all_nets_answers = np.zeros((len(net_ensemble), len(test_data)), dtype=int)
    idx = 0
    for name, net in net_ensemble:
        answers = net.answers(test_data)
        all_nets_answers[idx] = answers
        idx += 1

    # take majority vote
    correct = 0
    for i in range(len(all_nets_answers[0])):
        all_answers = {}
        majority = 0
        for j in range(len(all_nets_answers)):
            answer = all_nets_answers[j][i]
            try:
                all_answers[answer] += 1
            except KeyError:
                all_answers[answer] = 1

        values = all_answers.values()
        # if len(values) > 1:
        #     print("all options: ", all_answers, ", right answer: ",test_data[i][1])
        ans_majority = max(values)
        for key in all_answers:
            if all_answers[key] == ans_majority:
                majority = key
                break

        if majority == test_data[i][1]:
            correct += 1

    print(f"Majority scored {correct}/10000")
    return correct


# train_nets(networks,eta_vals, mini_batch_sizes, 30, r"C:\Users\Bryson M\Documents\USU\Classes\Intelligent Systems\hw03\pck_nets")
evaluate_net_ensemble(load_nets(r"C:\Users\Bryson M\Documents\USU\Classes\Intelligent Systems\hw03\pck_nets"), test_d)

# for name, net in load_nets("pck_nets"):
#     print(f"{name}")
#     print(net.evaluate(valid_d), "\n")