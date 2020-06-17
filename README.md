# StochasticProximalMethods


## Introduction

This package is the implementation of ProxSARAH algorithm and its variants along with other stochastic proximal gradient algorithms including ProxSVRG, ProxSpiderBoost, ProxSGD, and ProxGD to solve the stochastic composite, nonconvex, and possibly nonsmooth optimization problem which covers the composite finite-sum minimization problem as a special case.

## Code Usage

We hope that this program will be useful to others, and we would like to hear about your experience with it. If you found it helpful and are using it within our software please cite the following publication:

* N. H. Pham, L. M. Nguyen, D. T. Phan, and Q. Tran-Dinh. **[Proxsarah: An efficient algorithmic framework for stochastic composite non-convex optimization](http://jmlr.org/papers/v21/19-248.html)**. <em>Journal of Machine Learning Research</em>, 21(110):1–48,2020.

Feel free to send feedback and questions about the package to our maintainer Nhan H. Pham at <nhanph@live.unc.edu>.

## Code Organization

There are two sub-folders ``python_src`` and ``tensorflow_src`` containing the implementation of algorithms and examples in Python and Tensorflow, respectively. Please follow the instruction in the file ``README.md`` in each sub-folder on how to run each example.
