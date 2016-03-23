# -*-coding:utf-8-*-
#changed Multilayer Convolutional Network by yuma
#2015/12/29

#like python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#get mnist data
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#InteractiveSessionはグラフを作るときに柔軟に役立つらしい！
import tensorflow as tf
sess = tf.InteractiveSession()

class model:
    def __init__(self, shape):
        self.shape = shape
        self.input_data = tf.placeholder("float", shape = [None, 784])
        self.target     = tf.placeholder("float", shape = [None, 10])

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

    def conv2d(x, w):
        return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    def inference():
        self.w_conv1    = weight_variable([5, 5, 1, 32])
        self.b_conv1    = bias_variable([32])

################
shape = 784

mcn = model()

