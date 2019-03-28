"""!@package load_data

Useful function to read different dataset available from keras.

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
import keras
from keras.datasets import fashion_mnist, mnist, cifar10, cifar100
from keras import backend as K

def load_data(data_option):
	"""! Load dataset

	Depending on the name of dataset, this function will return the corresponding input.

	Parameters
	----------
	@param data_option : name of the dataset
	    
	Returns
	-------
	@retval x_train : train data
	@retval y_train : train label
	@retval x_test : test data
	@retval y_test : test label
	"""
	if data_option == 'mnist':
		# number of classes
		num_classes = 10

		# input image dimensions
		img_rows, img_cols, num_channel = 28, 28, 1

		# the data, shuffled and split between train and test sets
		(x_train, y_train), (x_test, y_test) = mnist.load_data()

		# reshape input data
		if K.image_data_format() == 'channels_first':
		    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
		    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
		    input_shape = (1, img_rows, img_cols)
		else:
		    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
		    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
		    input_shape = (img_rows, img_cols, 1)

		# set type as float
		x_train = x_train.astype('float32')
		x_test = x_test.astype('float32')

		# convert class vectors to binary class matrices
		y_train = keras.utils.to_categorical(y_train, num_classes)
		y_test = keras.utils.to_categorical(y_test, num_classes)

	elif data_option == 'fashion_mnist':
		# number of classes
		num_classes = 10

		# input image dimensions
		img_rows, img_cols, num_channel = 28, 28, 1

		# the data, shuffled and split between train and test sets
		(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

		# reshape input data
		if K.image_data_format() == 'channels_first':
		    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
		    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
		    input_shape = (1, img_rows, img_cols)
		else:
		    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
		    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
		    input_shape = (img_rows, img_cols, 1)

		# set type as float
		x_train = x_train.astype('float32')
		x_test = x_test.astype('float32')

		# convert class vectors to binary class matrices
		y_train = keras.utils.to_categorical(y_train, num_classes)
		y_test = keras.utils.to_categorical(y_test, num_classes)

	elif data_option == 'cifar':
		# number of classes
		num_classes = 10

		# input image dimensions
		img_rows, img_cols, num_channel = 32, 32, 3

		# the data, shuffled and split between train and test sets
		(x_train, y_train), (x_test, y_test) = cifar10.load_data()

		# reshape input data
		if K.image_data_format() == 'channels_first':
		    x_train = x_train.reshape(x_train.shape[0], num_channel, img_rows, img_cols)
		    x_test = x_test.reshape(x_test.shape[0], num_channel, img_rows, img_cols)
		    input_shape = (1, img_rows, img_cols)
		else:
		    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, num_channel)
		    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, num_channel)
		    input_shape = (img_rows, img_cols, 1)

		# set type as float
		x_train = x_train.astype('float32')
		x_test = x_test.astype('float32')

		# convert class vectors to binary class matrices
		y_train = keras.utils.to_categorical(y_train, num_classes)
		y_test = keras.utils.to_categorical(y_test, num_classes)

	return x_train, y_train, x_test, y_test