from __future__ import print_function

import math
import tensorflow as tf

def conv2d(x, W, b, strides=1):
  x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
  x = tf.nn.bias_add(x, b)
  return tf.nn.relu(x)

def conv_net(x, filter_sizes, filter_insizes, filter_strides, out_size):
  for layer_num in range(len(filter_sizes)):
    size = filter_sizes[layer_num]
    insize = filter_insizes[layer_num]

    if layer_num < len(filter_sizes) - 1:
      outsize = filter_insizes[layer_num + 1]
    else:
      outsize = out_size

    stride = filter_strides[layer_num]

    # Kernel and biases
    k1 = tf.Variable(tf.random_normal([size, size, insize, outsize],
    stddev = 0.01))
    b1 = tf.Variable(tf.zeros([outsize]))

    x = conv2d(x, k1, b1, stride)
  return x