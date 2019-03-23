"""!@package util_NonNegPCA

Useful functions for non-negative PCA example.

\f$ \min_{w\in\mathbb{R}^d}\left\{ f(w) := -\frac{1}{2n}\sum_{i=1}^nw^{\top}(z_iz_i^{\top})w \mid \|w\| \leq 1, ~w \geq 0 \right\} \f$

The package contains differnt functions to evaluate objective value, gradient as well as proximal operator for the non-negative PCA example.
"""

import numpy as np
import scipy
import random
import math

import time

## constant indicating total available memory when calculating full gradient
total_mem_full = 3.0e10

## constant indicating total available memory when calculating batch gradient
total_mem_batch = 2.0e10

def prox_half_l2_ball(w, lamb):
	"""! Compute the proximal operator of the indicator function of a half-l2 norm ball.

	\f$ prox_{\lambda \delta_{\mathcal{X}}(.)} = proj_{\mathcal{X}} \f$

	where $\mathcal{X} = \left\{w:~\|w\| \le 1,~w \ge 0 \right\}
	
	Parameters
	----------
	@param w : input vector
	@param lamb : penalty paramemeter, unused in this example
	    
	Returns
	-------
	@return perform projection onto half-l2 ball
	  
	"""
	mw = np.maximum(w,0)
	norm_mw = np.linalg.norm(mw, ord = 2)
	return mw / np.maximum(1,norm_mw)

def func_val_indicator(w):
	"""! Compute function value of indicator function \f$ \delta_{\mathcal{X}}(w) \f$.

	Parameters
	----------
	@param w : input vector

	Returns
	-------
	@return \f$ \delta_{\mathcal{X}}(w) \f$
	"""
	return 0

###################################################################

def func_val_non_neg_pca(n, Xw):
	"""! Compute the objective value

	\f$f(w) := -\frac{1}{2n}\sum_{i=1}^nw^{\top}(z_iz_i^{\top})w = -\frac{1}{2n}\sum_{i=1}^n(Xw)^{\top}(Xw) \f$

	@note The value of \f$z_i\f$ corresponds to row \f$i\f$ of \f$X\f$.

	Parameters
	----------
	@param n : sample size
	@param Xw : the precomputed \f$Xw\f$

	Returns
	-------
	@return \f$f(w)\f$
	"""
	return -(1.0/(2.0*float(n)))*np.dot(Xw, Xw)

""" Compute stochastic gradient / full gradient

Parameters
----------
n : int
	sample size
b : int
	batch size
	b = 1 - single stochastic gradient
	b = 2 - mini-batch
	b = n - full gradient
X : matrix
    input data
Y : array
	input label
bias : array
	input bias
w : 

Returns
-------
double
    objective value
"""
def grad_eval_non_neg_pca(n, d, b, X, Y, bias, w, nnzX = 0):
	"""! Compute the (full/stochastic) gradient.

	\f$f(w) := -\frac{1}{2n}\sum_{i=1}^nw^{\top}(z_iz_i^{\top})w = -\frac{1}{2n}\sum_{i=1}^n(Xw)^{\top}(Xw) \f$

	@note The value of \f$z_i\f$ corresponds to row \f$i\f$ of \f$X\f$.

	Parameters
	----------
	@param n : sample size
	@param d : number of features
	@param b : mini-batch size

		b = 1: single stochastic gradient

		1 < b < n: mini-batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label, unused in this example
	@param bias : input bias
	@param w : input vector
	@param nnzX : average number of non-zero elements for each sample

	Returns
	-------
	@return computed full/stochastic gradient

	@retval Xw: The precomputed \f$ Xw\fk since we do not have Y and bias in this example
	"""
	# single sample
	if b == 1:
		# get a random sample
		i = np.random.randint(0,n)

		z = X[i,:]

		return -z.T.dot(z.dot(w))
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

			batch_grad -= batch_X.transpose().dot(batch_X.dot(w))
        
		return batch_grad / float(b)

	else:
		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum(int(total_mem_full // nnzX), 1)
		num_batches = math.ceil(b / batch_size)
		full_grad = np.zeros(d)
		Xw = np.zeros(n)

		for j in range(num_batches): 
			# calculate start/end indices for each batch
			startIdx = batch_size*j
			endIdx = np.minimum(batch_size*(j+1), n-1)

			batch_X = X[startIdx:endIdx,:]

			batch_Xw = (batch_X.dot(w))

			Xw[startIdx:endIdx] = batch_Xw

			full_grad -= batch_X.transpose().dot(batch_Xw)

		return full_grad / float(n), Xw

def grad_diff_eval_non_neg_pca(n, d, b, X, Y, bias, w1, w2, nnzX = 0):
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
	@return computed full/stochastic gradient
	"""
	# single sample
	if b == 1:
		# get a random sample
		i = np.random.randint(0,n)

		z = X[i,:]

		return -z.T.dot(z.dot(w2 - w1))
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

			batch_grad_diff -= batch_X.transpose().dot(batch_X.dot(w2 - w1))
        
		return batch_grad_diff / float(b)

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

			batch_X = X[startIdx:endIdx,:]

			full_grad_diff -= batch_X.transpose().dot(batch_X.dot(w2 - w1))

		return full_grad_diff / float(n)
