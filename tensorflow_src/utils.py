"""!@package utils

Some useful functions used to define models and evaluate proximal operators.

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

def fc_relu(x, W, b):
    """! Fully connected layer

    Fully connected operation with ReLU as activation function.

    Parameters
    ----------
    @param x : placeholder for input data
    @param W : weight
    @param b : bias
        
    Returns
    -------
    @retval  : output of a MaxPooling2D layer
    """
    x = tf.add(tf.matmul(x, W), b)
    return tf.nn.relu(x)

def prox_l1(w, lamb):
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
    return tf.multiply(tf.sign(w),tf.maximum(tf.abs(w) - lamb,0) )
