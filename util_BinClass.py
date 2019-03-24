"""!@package util_BinClass

Useful functions for binary classification with non-convex loss example.

\f$\min_{w\in\mathbb{R}^d}\left\{ F(w) := \frac{1}{n}\sum_{i=1}^n\ell(a_i^{\top}w, b_i) + \lambda\|w\|_1\right\} \f$

The package contains differnt functions to evaluate objective value, gradient as well as proximal operator for the binary classification with
nonconvex loss example.

Copyright (c) 2019 Nhan H. Pham, Department of Statistics and Operations Research, University of North Carolina at Chapel Hill

Copyright (c) 2019 Quoc Tran-Dinh, Department of Statistics and Operations Research, University of North Carolina at Chapel Hill

Copyright (c) 2019 Lam M. Nguyen, IBM Research, Thomas J. Watson Research Center
Yorktown Heights

Copyright (c) 2019 Dzung T. Phan, IBM Research, Thomas J. Watson Research Center
Yorktown Heights
All rights reserved.

If you found this helpful and are using it within our software please cite the following publication:

* N. H. Pham, L. M. Nguyen, D. T. Phan, and Q. Tran-Dinh, **[ProxSARAH: An Efficient Algorithmic Framework for Stochastic Composite Nonconvex Optimization](https://arxiv.org/abs/1902.05679)**, _Arxiv preprint_, 2019.

"""

# external library
import numpy as np
import scipy
import random
import math

## constant indicating total available memory when calculating full gradient
total_mem_full = 3.0e10

## constant indicating total available memory when calculating batch gradient
total_mem_batch = 2.0e10


def prox_l1_norm(w, lamb):
	"""! Compute the proximal operator of the \f$\ell_1\f$-norm

	\f$ prox_{\lambda \|.\|_1} = {arg\min_x}\left\{\|.\|_1^2 + \frac{1}{2\lambda}\|x - w\|^2\right\} \f$
	
	Parameters
	----------
	@param w : input vector
	@param lamb : penalty paramemeter
	    
	Returns
	-------
	@retval : perform soft-thresholding on input vector
	"""
	return np.sign(w) * np.maximum( np.abs(w) - lamb, 0)

def func_val_l1_norm(w):
	"""! Compute \f$\ell_1\f$-norm of a vector

	Parameters
	----------
	@param w : input vector

	Returns
	-------
	@retval : \f$ \|w\|_1 \f$
	"""
	return np.linalg.norm(w,ord = 1)

def accuracy(n, d, X, Y, bias, w, nnzX = 0):
	"""! Compute accuracy

	Parameters
	----------
	@param n : sample size
	@param d : number of features
	@param X : input data
	@param Y : input label
	@param bias : bias vector
	@param w : input vector
	@param nnzX : average number of non-zero elements for each sample

	Returns
	-------
	@retval : value between 0-1 indicating the accuracy
	"""
	if nnzX == 0:
		nnzX = d
	batch_size = np.maximum(int(total_mem_full // nnzX), 1)
	num_batches = math.ceil(n / batch_size)
	sum_acc = 0

	for j in range(num_batches): 
		## calculate start/end indices for each batch
		startIdx = batch_size*j
		endIdx = np.minimum(batch_size*(j+1), n)

		batch_X = X[startIdx:endIdx]
		batch_Y = Y[startIdx:endIdx]
		batch_bias = bias[startIdx:endIdx]

		sum_acc += np.sum(1 * (batch_Y*(batch_X.dot(w) + batch_bias) > 0))

	return 1/float(n) * sum_acc

###################################################################

def func_val_bin_class_loss_1(n, XYw_bias):
	"""! Compute the objective value of loss function 1

	\f$\ell_1(Y(Xw+b)) := 1 - \tanh(\omega Y(Xw+b)) \f$

	for a given \f$ \omega > 0\f$.

	Parameters
	----------
	@param n : sample size
	@param Xw_bias : the precomputed \f$Y(Xw + b)\f$

	Returns
	-------
	@retval : \f$\ell_1(Y(Xw + b))\f$
	"""
	omega = 1.0
	expt = np.exp(2.0*omega*XYw_bias)
	return (1.0/float(n)) * np.sum( 2.0 / (expt + 1.0) )

def grad_eval_bin_class_loss_1(n, d, b, X, Y, bias, w, nnzX = 0):
	"""! Compute the (full/stochastic) gradient of loss function 1.

	where \f$\ell_1(Y(Xw+b)) := 1 - \tanh(\omega Y(Xw+b)) \f$

	for a given \f$ \omega > 0\f$.

	Parameters
	----------
	@param n : sample size
	@param d : number of features
	@param b : mini-batch size

		b = 1: single stochastic gradient

		1 < b < n: mini-batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label
	@param bias : input bias
	@param w : input vector
	@param nnzX : average number of non-zero elements for each sample

	Returns
	-------
	@retval : computed full/stochastic gradient

	@retval XYw_bias: The precomputed \f$ Y(Xw + bias)\f$
	"""
	omega = 1.0
	# single sample
	if b == 1:
		# get a random sample
		i = np.random.randint(0, n)

		Xi = X[i,:]
		expt = np.exp( 2.0*omega*Y[i]*(Xi.dot(w) + bias[i]) )

		return -4.0*omega*( expt/(expt + 1.0)/(expt + 1.0) )*Y[i]*Xi
	# batch
	elif b < n:
		# get a random batch of size b
		index = random.sample(range(n), b)

		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum(int(total_mem_full // nnzX), 1)
		num_batches = math.ceil(b / batch_size)
		batch_grad = np.zeros(d)

		for j in range(num_batches): 
			# calculate start/end indices for each batch
			startIdx = batch_size*j
			endIdx = np.minimum(batch_size*(j+1), b-1)

			batch_X = X[index[startIdx:endIdx],:]
			batch_Y = Y[index[startIdx:endIdx]]
			batch_bias = bias[index[startIdx:endIdx]]

			expt = np.exp( 2.0*omega * batch_Y * (batch_X.dot(w) + batch_bias) )

			batch_grad -= 4.0 * omega * batch_X.transpose().dot(batch_Y*(expt/(expt + 1.0)/(expt + 1.0))) 

		return batch_grad / float(b)
	# full
	else:
		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum(int(total_mem_full // nnzX), 1)
		num_batches = math.ceil(b / batch_size)
		full_grad = np.zeros(d)
		XYw_bias = np.zeros(n)

		for j in range(num_batches): 
			# calculate start/end indices for each batch
			startIdx = batch_size*j
			endIdx = np.minimum(batch_size*(j+1), n-1)

			batch_X = X[startIdx:endIdx,:]
			batch_Y = Y[startIdx:endIdx]
			batch_bias = bias[startIdx:endIdx]

			batch_XYw_bias = batch_Y * (batch_X.dot(w) + batch_bias)

			XYw_bias[startIdx:endIdx] = batch_XYw_bias

			expt = np.exp(2.0*omega * batch_XYw_bias)

			full_grad -= 4.0 * omega * batch_X.transpose().dot(batch_Y*(expt/(expt + 1.0)/(expt + 1.0))) 

		return full_grad / float(n), XYw_bias

def grad_diff_eval_bin_class_loss_1(n, d, b, X, Y, bias, w1, w2, nnzX = 0):
	"""! Compute the (full/stochastic) gradient difference of loss function 1

	\f$\displaystyle\frac{1}{b}\left(\sum_{i \in \mathcal{B}_t}(\nabla f_i(w_2) - \nabla f_i(w_1)) \right) \f$

	Parameters
	----------
	@param n : sample size
	@param d : number of features
	@param b : mini-batch size

		b = 1: single stochastic gradient

		1 < b < n: mini-batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label
	@param bias : input bias
	@param w1 : input vector
	@param w2 : input vector
	@param nnzX : average number of non-zero elements for each sample

	Returns
	-------
	@retval  : computed full/stochastic gradient
	"""
	omega = 1.0
	# single sample
	if b == 1:
		# get a random sample
		i = np.random.randint(0,n)

		Xi = X[i, :]
		expt1 = np.exp( 2.0*omega * Y[i] * (Xi.dot(w1) + bias[i]) )
		expt2 = np.exp( 2.0*omega * Y[i] * (Xi.dot(w2) + bias[i]) )

		diff_expt = expt2 / (expt2 + 1.0) / (expt2 + 1.0) - expt1 / (expt1 + 1.0) / (expt1 + 1.0)
		
		return -(4.0 * omega * diff_expt * Y[i]) * Xi
	# batch
	elif b < n:
		# get a random batch of size b
		index = random.sample(range(n), b)

		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum(int(total_mem_full // nnzX), 1)
		num_batches = math.ceil(b / batch_size)
		batch_grad_diff = np.zeros(d)

		for j in range(num_batches): 
			# calculate start/end indices for each batch
			startIdx = batch_size*j
			endIdx = np.minimum(batch_size*(j+1), b-1)

			batch_X = X[index[startIdx:endIdx],:]
			batch_Y = Y[index[startIdx:endIdx]]
			batch_bias = bias[index[startIdx:endIdx]]

			expt1 = np.exp( 2.0*omega * batch_Y * (batch_X.dot(w1) + batch_bias) )
			expt2 = np.exp( 2.0*omega * batch_Y * (batch_X.dot(w2) + batch_bias) )

			diff_expt = expt2/(expt2 + 1.0)/(expt2 + 1.0) - expt1/(expt1 + 1.0)/(expt1 + 1.0)

			batch_grad_diff -= 4.0 * omega * batch_X.transpose().dot(batch_Y * diff_expt )

		return batch_grad_diff / float(b)
	# full
	else:
		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum(int(total_mem_full // nnzX), 1)
		num_batches = math.ceil(b / batch_size)
		full_grad_diff = np.zeros(d)

		for j in range(num_batches): 
			# calculate start/end indices for each batch
			startIdx = batch_size*j
			endIdx = np.minimum(batch_size*(j+1), n-1)

			batch_X = X[startIdx:endIdx]
			batch_Y = Y[startIdx:endIdx]
			batch_bias = bias[startIdx:endIdx]

			expt1 = np.exp( 2.0*omega * batch_Y * (batch_X.dot(w1) + batch_XYw_bias) )
			expt2 = np.exp( 2.0*omega * batch_Y * (batch_X.dot(w2) + batch_XYw_bias) )

			diff_expt = expt2/(expt2 + 1.0)/(expt2 + 1.0) - expt1/(expt1 + 1.0)/(expt1 + 1.0)

			full_grad_diff -= 4.0 * omega * batch_X.transpose().dot(batch_Y * diff_expt )

		return full_grad_diff / float(n)

######################################################################

def func_val_bin_class_loss_2(n, XYw_bias):
	"""! Compute the objective value of loss function 2

	\f$\ell_2(Y(Xw+b)) := \left(1 - \frac{1}{1 + \exp[-Y(Xw+b)]}\right)^2 \f$

	for a given \f$ \omega > 0\f$.

	Parameters
	----------
	@param n : sample size
	@param Xw_bias : the precomputed \f$Y(Xw + b)\f$

	Returns
	-------
	@retval  : \f$\ell_2(Y(Xw + b))\f$
	"""
	expt = np.exp( XYw_bias )
	return (1.0/float(n))*np.sum ( 1.0 / ( (expt + 1.0)**2.0 ) )

def grad_eval_bin_class_loss_2(n, d, b, X, Y, bias, w, nnzX = 0):
	"""! Compute the (full/stochastic) gradient of loss function 2.

	\f$\ell_2(Y(Xw+b)) := \left(1 - \frac{1}{1 + \exp[-Y(Xw+b)]}\right)^2 \f$

	Parameters
	----------
	@param n : sample size
	@param d : number of features
	@param b : mini-batch size

		b = 1: single stochastic gradient

		1 < b < n: mini-batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label
	@param bias : input bias
	@param w : input vector
	@param nnzX : average number of non-zero elements for each sample

	Returns
	-------
	@retval  : computed full/stochastic gradient

	@retval XYw_bias: The precomputed \f$ Y(Xw + bias)\f$
	"""
	# single sample
	if b == 1:
		# get a random sample
		i = np.random.randint(0, n)
		
		Xi = X[i, :]
		expt = np.exp(Y[i] * (Xi.dot(w) + bias[i]) )
		
		return ( -2.0 * (expt/(1.0 + expt)/(1.0 + expt)/(1.0 + expt)) * Y[i] ) * Xi
	# batch
	elif b < n:
		# get a random batch of size b
		index = random.sample(range(n), b)

		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum(int(total_mem_full // nnzX), 1)
		num_batches = math.ceil(b / batch_size)
		batch_grad = np.zeros(d)

		for j in range(num_batches): 
			# calculate start/end indices for each batch
			startIdx = batch_size*j
			endIdx = np.minimum(batch_size*(j+1), b-1)

			batch_X = X[index[startIdx:endIdx],:]
			batch_Y = Y[index[startIdx:endIdx]]
			batch_bias = bias[index[startIdx:endIdx]]

			expt = np.exp( batch_Y * (batch_X.dot(w) + batch_bias) )

			batch_grad -= 2.0 * batch_X.transpose().dot(batch_Y \
											* (expt/(1.0 + expt)/(1.0 + expt)/(1.0 + expt)) )
        
		return batch_grad / float(b)
	# full
	else:
		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum(int(total_mem_full // nnzX), 1)
		num_batches = math.ceil(b / batch_size)
		full_grad = np.zeros(d)
		XYw_bias = np.zeros(n)

		for j in range(num_batches): 
			# calculate start/end indices for each batch
			startIdx = batch_size*j
			endIdx = np.minimum(batch_size*(j+1), n-1)

			batch_X = X[startIdx:endIdx]
			batch_Y = Y[startIdx:endIdx]
			batch_bias = bias[startIdx:endIdx]

			batch_XYw_bias = batch_Y * (batch_X.dot(w) + batch_bias)

			XYw_bias[startIdx:endIdx] = batch_XYw_bias

			expt = np.exp(batch_XYw_bias)

			full_grad -= 2.0 * batch_X.transpose().dot( batch_Y * (expt/(expt+1)/(expt+1)/(expt+1) ) )

		return full_grad / float(n), XYw_bias

def grad_diff_eval_bin_class_loss_2(n, d, b, X, Y, bias, w1, w2, nnzX = 0):
	"""! Compute the (full/stochastic) gradient difference of loss function 2

	\f$\displaystyle\frac{1}{b}\left(\sum_{i \in \mathcal{B}_t}(\nabla f_i(w_2) - \nabla f_i(w_1)) \right) \f$

	Parameters
	----------
	@param n : sample size
	@param d : number of features
	@param b : mini-batch size

		b = 1: single stochastic gradient

		1 < b < n: mini-batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label
	@param bias : input bias
	@param w1 : input vector
	@param w2 : input vector
	@param nnzX : average number of non-zero elements for each sample

	Returns
	-------
	@retval  : computed full/stochastic gradient
	"""
	# single sample
	if b == 1:
		# get a random sample
		i = np.random.randint(0, n)
		
		Xi = X[i,:]
		expt1 = np.exp(Y[i]* (Xi.dot(w1) + bias[i]) )
		expt2 = np.exp(Y[i] * (Xi.dot(w2) + bias[i]))

		diff_expt = (expt2/(1.0 + expt2)/(1.0 + expt2)/(1.0 + expt2)) - (expt1/(1.0 + expt1)/(1.0 + expt1)/(1.0 + expt1))
		
		return -2.0*diff_expt*Y[i]*Xi
	# batch
	elif b < n:
		# get a random batch of size b
		index = random.sample(range(n), b)

		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum(int(total_mem_full // nnzX), 1)
		num_batches = math.ceil(b / batch_size)
		batch_grad_diff = np.zeros(d)

		for j in range(num_batches): 
			# calculate start/end indices for each batch
			startIdx = batch_size*j
			endIdx = np.minimum(batch_size*(j+1), b-1)

			batch_X = X[index[startIdx:endIdx],:]
			batch_Y = Y[index[startIdx:endIdx]]
			batch_bias = bias[index[startIdx:endIdx]]

			expt1 = np.exp(batch_Y * (batch_X.dot(w1) + batch_bias) )
			expt2 = np.exp(batch_Y * (batch_X.dot(w2) + batch_bias) )

			diff_expt = expt2/(1.0 + expt2)/(1.0 + expt2)/(1.0 + expt2) - expt1/(1.0 + expt1)/(1.0 + expt1)/(1.0 + expt1)
		
			batch_grad_diff -= 2.0 * batch_X.transpose().dot( batch_Y * diff_expt )

		return batch_grad_diff / float(b)
	# full
	else:
		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum(int(total_mem_full // nnzX), 1)
		num_batches = math.ceil(b / batch_size)
		full_grad_diff = np.zeros(d)

		for j in range(num_batches): 
			# calculate start/end indices for each batch
			startIdx = batch_size*j
			endIdx = np.minimum(batch_size*(j+1), n-1)

			batch_X = X[startIdx:endIdx]
			batch_Y = Y[startIdx:endIdx]
			batch_bias = bias[startIdx:endIdx]

			expt1 = np.exp(batch_Y * (batch_X.dot(w1) + batch_bias) )
			expt2 = np.exp(batch_Y * (batch_X.dot(w2) + batch_bias) )

			expt1 = np.exp( 2.0*omega * batch_Y * (batch_X.dot(w1) + batch_XYw_bias) )
			expt2 = np.exp( 2.0*omega * batch_Y * (batch_X.dot(w2) + batch_XYw_bias) )
			diff_expt = expt2/(1.0 + expt2)/(1.0 + expt2)/(1.0 + expt2) - expt1/(1.0 + expt1)/(1.0 + expt1)/(1.0 + expt1)

			full_grad_diff -= 2.0 * batch_X.transpose().dot( batch_Y * diff_expt )

		return full_grad_diff / float(n)

##################################################################

def func_val_bin_class_loss_3(n, XYw_bias):
	"""! Compute the objective value of loss function 3

	\f$ \ell_3(Y(Xw + b)) := \ln(1 + \exp(-Y(Xw + b))) - \ln(1 + \exp(-Y(Xw + b) - \omega))\f$

	for a given \f$ \omega > 0\f$.

	Parameters
	----------
	@param n : sample size
	@param Xw_bias : the precomputed \f$Y(Xw + b)\f$

	Returns
	-------
	@retval : \f$\ell_3(Y(Xw + b))\f$
	"""
	omega = 1.0
	exp_g = np.exp(-omega)
	expt = np.exp(-XYw_bias)

	return (1.0 / float(n)) * np.sum((np.log(1.0 + expt) - np.log(1.0 + exp_g*expt)))

def grad_eval_bin_class_loss_3(n, d, b, X, Y, bias, w, nnzX = 0):
	"""! Compute the (full/stochastic) gradient of loss function 3.

	where \f$ \ell_3(Y(Xw + b)) := \ln(1 + \exp(-Y(Xw + b))) - \ln(1 + \exp(-Y(Xw + b) - \omega))\f$

	for a given \f$ \omega > 0\f$.

	Parameters
	----------
	@param n : sample size
	@param d : number of features
	@param b : mini-batch size

		b = 1: single stochastic gradient

		1 < b < n: mini-batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label
	@param bias : input bias
	@param w : input vector
	@param nnzX : average number of non-zero elements for each sample

	Returns
	-------
	@retval  : computed full/stochastic gradient

	@retval XYw_bias: The precomputed \f$ Y(Xw + bias)\f$
	"""
	alpha = 1
	exp_a = np.exp(alpha)
	# single sample
	if b == 1:
		# get a random sample
		i = np.random.randint(0, n)

		Xi = X[i, :]
		expt = np.exp( Y[i] * (Xi.dot(w) + bias[i]) )

		return ( (1 / (expt * exp_a + 1.0) - 1 / (expt + 1.0) ) * Y[i] ) * Xi
	# batch
	elif b < n:
		# get a random batch of size b
		index = random.sample(range(n), b)

		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum(int(total_mem_full // nnzX), 1)
		num_batches = math.ceil(b / batch_size)
		batch_grad = np.zeros(d)

		for j in range(num_batches): 
			# calculate start/end indices for each batch
			startIdx = batch_size*j
			endIdx = np.minimum(batch_size*(j+1), b-1)

			batch_X = X[index[startIdx:endIdx],:]
			batch_Y = Y[index[startIdx:endIdx]]
			batch_bias = bias[index[startIdx:endIdx]]

			expt = np.exp( batch_Y * (batch_X.dot(w) + batch_bias) )

			batch_grad += batch_X.transpose().dot(batch_Y * (1 / (expt * exp_a + 1.0) - 1 / (expt + 1.0)) )

		return batch_grad / float(b)
	# full
	else:
		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum(int(total_mem_full // nnzX), 1)
		num_batches = math.ceil(b / batch_size)
		full_grad = np.zeros(d)
		XYw_bias = np.zeros(n)

		for j in range(num_batches): 
			# calculate start/end indices for each batch
			startIdx = batch_size*j
			endIdx = np.minimum(batch_size*(j+1), n-1)

			batch_X = X[startIdx:endIdx,:]
			batch_Y = Y[startIdx:endIdx]
			batch_bias = bias[startIdx:endIdx]

			batch_XYw_bias = batch_Y * (batch_X.dot(w) + batch_bias)

			XYw_bias[startIdx:endIdx] = batch_XYw_bias

			expt = expt = np.exp(batch_XYw_bias)

			full_grad = batch_X.transpose().dot(batch_Y * ( 1.0/ (expt * exp_a + 1.0) - 1.0/ (expt + 1.0) ) )

		return full_grad / float(n), XYw_bias
		
def grad_diff_eval_bin_class_loss_3(n, d, b, X, Y, bias, w1, w2, nnzX = 0):
	"""! Compute the (full/stochastic) gradient difference of loss function 3

	\f$\displaystyle\frac{1}{b}\left(\sum_{i \in \mathcal{B}_t}(\nabla f_i(w_2) - \nabla f_i(w_1)) \right) \f$

	Parameters
	----------
	@param n : sample size
	@param d : number of features
	@param b : mini-batch size

		b = 1: single stochastic gradient

		1 < b < n: mini-batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label
	@param bias : input bias
	@param w1 : input vector
	@param w2 : input vector
	@param nnzX : average number of non-zero elements for each sample

	Returns
	-------
	@retval  : computed full/stochastic gradient
	"""
	alpha = 1
	exp_a = np.exp(alpha)
	# single sample
	if b == 1:
		# get a random sample
		i = np.random.randint(0, n)

		Xi = X[i,:]
		expt1 = np.exp( -Y[i]*(Xi.dot(w1) + bias[i]) )
		expt2 = np.exp( -Y[i]*(Xi.dot(w2) + bias[i]) )

		diff_expt = (1.0/(expt2*exp_a + 1.0) - 1.0/(expt2 + 1.0)) - (1.0/(expt1*exp_a + 1.0) - 1.0/(expt1 + 1.0))
		
		return diff_expt*Y[i]*Xi
	# batch
	elif b < n:
		# get a random batch of size b
		index = random.sample(range(n), b)

		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = int(total_mem_batch // nnzX)
		num_batches = math.ceil(b / batch_size)
		batch_grad_diff = np.zeros(d)

		for j in range(num_batches): 
			# calculate start/end indices for each batch
			startIdx = batch_size*j
			endIdx = np.minimum(batch_size*(j+1), b-1)

			batch_X = X[index[startIdx:endIdx],:]
			batch_Y = Y[index[startIdx:endIdx]]
			batch_bias = bias[index[startIdx:endIdx]]

			expt1 = np.exp(batch_Y * (batch_X.dot(w1) + batch_bias) )
			expt2 = np.exp(batch_Y * (batch_X.dot(w2) + batch_bias) )

			diff_expt = ( 1/(expt2*exp_a + 1.0) - 1/(expt2 + 1.0) ) - (1/(expt1*exp_a + 1.0) - 1/(expt1 + 1.0) )

			batch_grad_diff += batch_X.transpose().dot( batch_Y * diff_expt )

		return batch_grad_diff / float(b)
	# full
	else:
		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = int(total_mem_full // nnzX)
		num_batches = math.ceil(b / batch_size)
		full_grad_diff = np.zeros(d)

		for j in range(num_batches): 
			# calculate start/end indices for each batch
			startIdx = batch_size*j
			endIdx = np.minimum(batch_size*(j+1), n-1)

			batch_X = X[startIdx:endIdx]
			batch_Y = Y[startIdx:endIdx]
			batch_bias = bias[startIdx:endIdx]

			expt1 = np.exp(batch_Y * (batch_X.dot(w1) + batch_bias) )
			expt2 = np.exp(batch_Y * (batch_X.dot(w2) + batch_bias) )

			diff_expt = ( 1/(expt2*exp_a + 1.0) - 1/(expt2 + 1.0) ) - (1/(expt1*exp_a + 1.0) - 1/(expt1 + 1.0) )

			full_grad_diff += batch_X.transpose().dot( batch_Y * diff_expt )

		return full_grad_diff / float(n)
