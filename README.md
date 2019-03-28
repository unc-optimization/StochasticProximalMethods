# StochasticProximalMethods


## Introduction

This package is the implementation of ProxSARAH algorithm and its variants along with other stochastic proximal gradient algorithms including ProxSVRG, ProxSpiderBoost, ProxSGD, and ProxGD to solve the stochastic composite, nonconvex, and possibly nonsmooth optimization problem which covers the composite finite-sum minimization problem as a special case.

## Code Usage

We hope that this program will be useful to others, and we would like to hear about your experience with it. If you found it helpful and are using it within our software please cite the following publication:

* N. H. Pham, L. M. Nguyen, D. T. Phan, and Q. Tran-Dinh, **[ProxSARAH: An Efficient Algorithmic Framework for Stochastic Composite Nonconvex Optimization](https://arxiv.org/abs/1902.05679)**, _arXiv preprint arXiv:1902.05679_, 2019.

Feel free to send feedback and questions about the package to

## Code Organization

There are two sub-folders ``python_src`` and ``tensorflow_src`` containing the implementation of algorithms and examples in Python and Tensorflow, respectively. Please follow the instruction in the file ``README.md`` in each sub-folder on how to run each example.