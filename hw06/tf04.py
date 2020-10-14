###########################################
# module: tf04.py
# description: tf tensorflow examples
# bugs to vladimir dot kulyukin via canvas
###########################################

import tensorflow as tf
import numpy as np

## A, B, C are arrays.
A = tf.placeholder(tf.float32, [None, 1], name='A')
B = tf.placeholder(tf.float32, [None, 1], name='B')
C = tf.placeholder(tf.float32, [None, 1], name='C')
D = tf.constant(5.0, name='D')

## Let's define a few tensor operations.
E = tf.add(A, B, name='E') ## 0+2 = 2
F = tf.subtract(B, C, name='F') ## 2-4 = -2
G = tf.multiply(E, F, name='G') ## 2(-2) = -4
H = tf.subtract(G, D, name='H') ## -4 - 5 = -9

## Let's define tf initialize operation.
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # initialize the variables
    sess.run(init_op)
    # compute the output of the graph
    # note that these values are computed outside
    # of the dictionary before it's defined.
    AV = np.array([0, 0, 1, 1, 2])[:,np.newaxis]
    BV = np.array([2, 2, 3, 3, 4])[:,np.newaxis]
    CV = np.array([4, 4, 5, 5, 6])[:,np.newaxis]
    HV = sess.run(H,
                  feed_dict={A: AV,
                             B: BV,
                             C: CV
                  })
    # print out the result.
    print('Variable H is {}'.format(HV))
