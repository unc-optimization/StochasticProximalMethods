# StochasticProximalMethods


## Introduction


This package is the implementation of ProxSARAH algorithm and its variants along with other stochastic proximal gradient algorithms including ProxSVRG, ProxSpiderBoost, ProxSGD, and ProxGD to solve the stochastic composite, nonconvex, and possibly nonsmooth optimization problem which covers the composite finite-sum minimization problem as a special case.

## Code Usage

We hope that this program will be useful to others, and we would like to hear about your experience with it. If you found it helpful and are using it within our software please cite the following publication:

* N. H. Pham, L. M. Nguyen, D. T. Phan, and Q. Tran-Dinh, **[ProxSARAH: An Efficient Algorithmic Framework for Stochastic Composite Nonconvex Optimization](https://arxiv.org/abs/1902.05679)**, _arXiv preprint arXiv:1902.05679_, 2019.

Feel free to send feedback and questions about the package to our maintainer Nhan H. Pham
<nhanph@live.unc.edu>.

## Dependencies

The code is tested under Python 3.6.3 and it requires additional packages if you do not have them

* scipy: for working with datasets
* argParser: for argument parsing
* matplotlib: for plotting
* sklearn: for normalizing input data
* joblib: for caching the dataset

```
pip install scipy argParser matplotlib sklearn joblib
```

The package supports LIBSVM dataset which can be downloaded [here](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html).

## How to run

1. Modify the path to your dataset folder:
	In order to run the code, you need to provide a folder that contain your dataset. To do so, go to `import_data.py` script and modify the `data_path` variable.

	* Example:
	```python
	data_path = '/home/MyPC/dataset'
	```

2. Understanding the argument:
	There are several arguments needed to run the script for each example. The main ones include

| Argument     | Description                        |
| -------------|:----------------------------------:| 
| -h           | print help message                 |
| -a           | select algorithm: from 1 to 6      |
| -so          | select ProxSARAH variants          | 
| -aso         | select ProxSARAH-Adaptive variants | 
| -ne          | number of total epochs to run      |

More information can be found by running the corresponding example script with option -h
```python
python non_neg_pca_example.py -h

python binary_classification_example.py -h
```

2. If you want to run the Nonnegative PCA example, use the command below

* single sample case:
```python
python non_neg_pca_example.py -d mnist -a 12456
```
which means we are running NonNegative PCA example with dataset `mnist` using 5 algorithms: ProxSARAH-v1, ProxSARAH-A-v1, ProxSVRG, ProxSGD, and ProxGD, respectively. You can run a subset of the algorithms too:
```python
python non_neg_pca_example.py -d mnist -a 124
```
only runs ProxSARAH-v1, ProxSARAH-A-v1, and ProxSVRG.

* mini batch:
```python
python non_neg_pca_example.py -d mnist -a 123456 -b 250 -so 2345 -aso 23
```
**Meaning:** we are running NonNegative PCA example with dataset `mnist` using 6 algorithms: ProxSARAH-v2 to ProxSARAH-v5, ProxSARAH-A-v2, ProxSARAH-A-v3, ProxSpiderBoost, ProxSVRG, ProxSGD, and ProxGD, respectively. Here, the batchsize is specified through option `-b 250`. **Important: You must specified a batch size > 1 in this case.**

3. If you want to run the Binary Classification with nonconvex loss example, use the command below

* single sample case:
```python
python binary_classification_example.py -d news20.binary -a 12456
```

* mini batch:
```python
python binary_classification_example.py -d news20.binary -a 123456 -b 200 -so 2345 -aso 23
```
The interpretation for each argument is the same as in nonnegative PCA example.