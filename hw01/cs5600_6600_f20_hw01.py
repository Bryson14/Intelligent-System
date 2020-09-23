
####################################################
# CS 5600/6600: F20: Assignment 1
# Bryson Meiling
# A01503163
#####################################################

import numpy as np

'''
For every function "output", the parameter x is assumed to be a numpy array
'''
class and_percep:

    def __init__(self):
        w_0 = .5
        w_1 = .5
        self.b = 1
        self.weights = np.array([w_0, w_1])
        
    def output(self, x):
        assert isinstance(x, np.ndarray) and len(x) == 2
        return int(self.weights.dot(x) >= self.b)

class or_percep:
    
    def __init__(self):
        w_0 = 1
        w_1 = 1
        self.b = 1
        self.weights = np.array([w_0, w_1])

    def output(self, x):
        assert isinstance(x, np.ndarray) and len(x) == 2
        return int(self.weights.dot(x) >= self.b)

class not_percep:
    
    def __init__(self):
        self.w = np.array([1])
        self.b = 0

    def output(self, x):
        assert isinstance(x, np.ndarray) and len(x) == 1
        return int(self.w.dot(x) <= 0)

class xor_percep:
    '''
    o ----> OR \
        x          AND
    o ----> NAND /
    '''
    
    def __init__(self):
        self.notp = not_percep()
        self.orp = or_percep()
        self.andp = and_percep()

    def output(self, x):
        or_gate = self.orp.output(x)
        nand_gate = self.notp.output(np.array([self.andp.output(x)]))
        return self.andp.output(np.array([nand_gate, or_gate]))

class xor_percep2:
    # XOR gate in terms of weights and biases, not other perceptrons
    def __init__(self):
        self.b = 1
        self.or_w = np.array([1,1])  # or biases
        self.and_w = np.array([.5,.5])  # and biases for two D
        self.and_w_1 = np.array([.5])   # and bias for one D

    def threshold(self, x, y, b):
        return int(x.dot(y) >= b)
    
    def output(self, x):
        assert isinstance(x, np.ndarray) and len(x) == 2
        orp = np.array([self.threshold(x, self.or_w, 1)])
        nandp = np.array([int(self.threshold(x, self.and_w, 1) <= 0)])
        return np.array([self.threshold(orp, nandp, self.and_w_1)])

class percep_net:
    # simulate (((x_0 V x_1) ^ - x_2) V x_3
    # where V = or, ^ = and, - = not
    def __init__(self):
        self.notp = not_percep()
        self.andp = and_percep()
        self.orp = or_percep()

    def output(self, x):
        assert isinstance(x, np.ndarray) and len(x) == 4
        first = np.array([self.orp.output(x[:2])])
        second = np.array([self.andp.output(np.concatenate([first, np.array([self.notp.output(x[2:3])])]))])
        third = self.orp.output(np.concatenate([second, x[-1:]]))
        return third

    
        





