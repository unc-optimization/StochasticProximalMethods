"""!@package util_NonComposite

Some simple function to use in non-composite setting to solve the non-convex problem

\f$\min_{w\in\mathbb{R}^d}\left\{ F(w) := \frac{1}{n}\sum_{i=1}^n\ell(a_i^{\top}w, b_i) \f$

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

def prox_empty(w, lamb):
	"""! Empty function used for non-composite settings
	
	Parameters
	----------
	@param w : input vector
	@param lamb : penalty paramemeter
	    
	Returns
	-------
	@retval : return same input
	"""
	return w

def func_val_empty(w):
	"""! Empty regularlizer used for non-composite settings

	Parameters
	----------
	@param w : input vector

	Returns
	-------
	@retval : \f$ 0 \f$
	"""
	return 0
