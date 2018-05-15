#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 15:57:43 2018

@author: epark
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf
sess = tf.InteractiveSession()

#import utils.mnist_reader as mnist_reader
#X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
#X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

# import the fashion mnist data
from tensorflow.examples.tutorials.mnist import input_data
fashion_mnist = input_data.read_data_sets('data/fashion', one_hot=True)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def batch_norm(x, dim):
    mean, variance = tf.nn.moments(x, [0])
    return tf.nn.batch_normalization(x, mean, variance, tf.Variable(tf.zeros([dim])), tf.Variable(tf.ones([dim])), 1e-3)

# Initialize weights and biases
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

# Initialize the x and y values
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1,28,28,1])

# First convolution + maxpool
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# batch_norm
h_batch1 = batch_norm(h_pool1, 32)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64]) 

h_conv2 = tf.nn.relu(conv2d(h_batch1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# batch norm
h_batch2 = batch_norm(h_pool2, 64)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_batch2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

for i in range(20000):
    
  batch = fashion_mnist.train.next_batch(50)
  
  if i % 100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    
    print("step %d, training accuracy %g"%(i, train_accuracy))
    print("test accuracy %g"%accuracy.eval({x: fashion_mnist.test.images, y_: fashion_mnist.test.labels, keep_prob: 1.0}))
    
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
