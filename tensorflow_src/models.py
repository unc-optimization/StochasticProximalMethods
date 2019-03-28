"""!@package models

Several feedforawrd neural network models used in the paper.

We implement two fully connected models with size [InSize]x100x[OutSize] and [InSize]x800x[OutSize] where [InSize] and [OutSize] are input and output dimensions.

Copyright (c) 2019 Nhan H. Pham, Department of Statistics and Operations Research, University of North Carolina at Chapel Hill

Copyright (c) 2019 Quoc Tran-Dinh, Department of Statistics and Operations Research, University of North Carolina at Chapel Hill

Copyright (c) 2019 Lam M. Nguyen, IBM Research, Thomas J. Watson Research Center
Yorktown Heights

Copyright (c) 2019 Dzung T. Phan, IBM Research, Thomas J. Watson Research Center
Yorktown Heights
All rights reserved.

If you found this helpful and are using it within our software please cite the following publication:

* N. H. Pham, L. M. Nguyen, D. T. Phan, and Q. Tran-Dinh, **[ProxSARAH: An Efficient Algorithmic Framework for Stochastic Composite Nonconvex Optimization](https://arxiv.org/abs/1902.05679)**, _arXiv preprint arXiv:1902.05679_, 2019.

"""

import tensorflow as tf
from tensorflow.contrib.layers import flatten
from utils import *

#==================================================================================================================
# ================ Tensorflow Model =======================

def model1(x, input_size, output_size):

	"""! Fully connected model [InSize]x100x[OutSize]

	Implementation of a [InSize]x100x[OutSize] fully connected model.

	Parameters
	----------
	@param x : placeholder for input data
	@param input_size : size of input data
	@param output_size : size of output data
	    
	Returns
	-------
	@retval logits : output
	@retval logits_dup : a copy of output
	@retval w_list : trainable parameters
	@retval w_list_dup : a copy of trainable parameters
	"""

	#==================================================================================================================
	## model definition
	mu = 0
	sigma = 0.2
	weights = {
	    'wfc': tf.Variable(tf.truncated_normal(shape=(input_size,100), mean = mu, stddev = sigma, seed = 1)),
	    'out': tf.Variable(tf.truncated_normal(shape=(100,output_size), mean = mu, stddev = sigma, seed = 1))
	}

	biases = {
	    'bfc': tf.Variable(tf.zeros(100)),
	    'out': tf.Variable(tf.zeros(output_size))
	}

	# Flatten input.
	c_flat = flatten(x)

	# Layer 1: Fully Connected. Input = input_size. Output = 100.    
	# Activation.
	fc = fc_relu(c_flat, weights['wfc'], biases['bfc'])       

	# Layer 2: Fully Connected. Input = 100. Output = output_size.
	logits = tf.add(tf.matmul(fc, weights['out']), biases['out'])

	w_list = []
	for w,b in zip(weights, biases):
	    w_list.append(weights[w])
	    w_list.append(biases[b])


	#==================================================================================================================
	## duplicate the model used in ProxSVRG
	weights_dup = {
	    'wfc': tf.Variable(tf.truncated_normal(shape=(input_size,100), mean = mu, stddev = sigma, seed = 1)),
	    'out': tf.Variable(tf.truncated_normal(shape=(100,output_size), mean = mu, stddev = sigma, seed = 1))
	}

	biases_dup = {
	    'bfc': tf.Variable(tf.zeros(100)),
	    'out': tf.Variable(tf.zeros(output_size))
	}

	# Flatten input.
	c_flat_dup = flatten(x)

	# Layer 1: Fully Connected. Input = input_size. Output = 100.    
	# Activation.
	fc_dup = fc_relu(c_flat_dup, weights_dup['wfc'], biases_dup['bfc'])       

	# Layer 2: Fully Connected. Input = 100. Output = output_size.
	logits_dup = tf.add(tf.matmul(fc_dup, weights_dup['out']), biases_dup['out'])

	w_list_dup = []
	for w,b in zip(weights_dup, biases_dup):
	    w_list_dup.append(weights_dup[w])
	    w_list_dup.append(biases_dup[b])

	return logits, logits_dup, w_list, w_list_dup

def model2(x, input_size, output_size):

	"""! Fully connected model [InSize]x800x[OutSize]

	Implementation of a [InSize]x800x[OutSize] fully connected model.

	Parameters
	----------
	@param x : placeholder for input data
	@param input_size : size of input data
	@param output_size : size of output data
	    
	Returns
	-------
	@retval logits : output
	@retval logits_dup : a copy of output
	@retval w_list : trainable parameters
	@retval w_list_dup : a copy of trainable parameters
	"""

	#==================================================================================================================
	## model definition
	mu = 0
	sigma = 0.2
	weights = {
	    'wfc': tf.Variable(tf.truncated_normal(shape=(input_size,800), mean = mu, stddev = sigma, seed = 1)),
	    'out': tf.Variable(tf.truncated_normal(shape=(800,output_size), mean = mu, stddev = sigma, seed = 1))
	}

	biases = {
	    'bfc': tf.Variable(tf.zeros(800)),
	    'out': tf.Variable(tf.zeros(output_size))
	}

	# Flatten input.
	c_flat = flatten(x)

	# Layer 1: Fully Connected. Input = input_size. Output = 800.    
	# Activation.
	fc = fc_relu(c_flat, weights['wfc'], biases['bfc'])       

	# Layer 2: Fully Connected. Input = 800. Output = output_size.
	logits = tf.add(tf.matmul(fc, weights['out']), biases['out'])

	w_list = []
	for w,b in zip(weights, biases):
	    w_list.append(weights[w])
	    w_list.append(biases[b])


	#==================================================================================================================
	## duplicate the model used in ProxSVRG
	weights_dup = {
	    'wfc': tf.Variable(tf.truncated_normal(shape=(input_size,800), mean = mu, stddev = sigma, seed = 1)),
	    'out': tf.Variable(tf.truncated_normal(shape=(800,output_size), mean = mu, stddev = sigma, seed = 1))
	}

	biases_dup = {
	    'bfc': tf.Variable(tf.zeros(800)),
	    'out': tf.Variable(tf.zeros(output_size))
	}

	# Flatten input.
	c_flat_dup = flatten(x)

	# Layer 1: Fully Connected. Input = input_size. Output = 800.    
	# Activation.
	fc_dup = fc_relu(c_flat_dup, weights_dup['wfc'], biases_dup['bfc'])       

	# Layer 2: Fully Connected. Input = 800. Output = output_size.
	logits_dup = tf.add(tf.matmul(fc_dup, weights_dup['out']), biases_dup['out'])

	w_list_dup = []
	for w,b in zip(weights_dup, biases_dup):
	    w_list_dup.append(weights_dup[w])
	    w_list_dup.append(biases_dup[b])

	return logits, logits_dup, w_list, w_list_dup
