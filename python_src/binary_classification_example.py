"""@package binary_classification_example

This package implements the binary classification with nonconvex loss example.

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

import matplotlib.pyplot as plt
import sklearn
from scipy import sparse
from import_data import *
from argParser import *

# import methods
from method_ProxSARAH import *
from method_ProxSARAH_Adaptive import *
from method_ProxSpiderBoost import *
from method_ProxSVRG import *
from method_ProxSGD import *
from method_ProxGD import *

# import utility functions
from util_BinClass import *

import os
import time

## USAGE:

# single sample:
# python binary_classification_example.py -d [dataset] -p1 -a 12457

# mini batch:
# python binary_classification_example.py -d [dataset] -p1 -a 123456 -b [batch_size] -so 2345 -aso 23

# -so: Proxsarah option
# 	help="Select minibatch and step sizes:\n \
# 			0: all\
# 			1: b = 1\
# 			2: b = m = sqrt(n), gamma = 0.95\
# 			3: b = m = sqrt(n), gamma = 0.99\
# 			4: b = m = n^(1/3), gamma = 0.95\
# 			5: b = m = n^(1/3), gamma = 0.99\
# 			")

#-aso: Proxsarah adaptive option
# 	help="Select minibatch and step sizes: \
# 			0: all\
# 			1: b = 1\
# 			1: b = m = sqrt(n)\
# 			2: b = m = n^(1/3)\
# 			")

#=================================================================
#=======================  Preprocessing  =========================
#=================================================================

#========================= Read input ============================

# read arguments
data_name, prog_option, alg_list, prox_sarah_option, prox_sarah_adaptive_option = argParser()

# get program parameters:
plot_option 	= prog_option["PlotOption"]
batch_size 		= prog_option["BatchSize"]
max_num_epochs 	= prog_option["MaxNumEpoch"]
verbose			= prog_option["Verbose"]
log_enable		= prog_option["LogEnable"]

# load data
print('Load data', data_name)
X_train, Y_train, X_test, Y_test = import_data(data_name)

# convert to sparse matrix if necessary
if not sparse.isspmatrix_csr(X_train):
	X_train = sparse.csr_matrix(X_train)

# get size of data
num_train, total_dim = np.shape(X_train)

if num_train == 0:
	sys.exit("Input dataset not found!")

if data_name == 'avazu-app' or data_name == 'criteo.kaggle2014' or data_name =='kddb-raw':
	min_val = np.amin(Y_train)
	max_val = np.amax(Y_train)
	mid_val = (min_val + max_val) / 2.0

	Y_train = 1*(Y_train >= mid_val) - 1*(Y_train < mid_val)

# get size of data
if len(Y_test) > 0:
	num_test, total_dim_test = np.shape(X_test)
	# conver test data to sparse matrix if necessary
	if total_dim_test == total_dim:
		if not sparse.isspmatrix_csr(X_test):
			X_test = sparse.csr_matrix(X_test)
		# check label, if not valid, convert to -1,1
		min_val = np.amin(Y_test)
		max_val = np.amax(Y_test)
		mid_val = (min_val + max_val) / 2.0

		Y_test = 1*(Y_test >= mid_val) - 1*(Y_test < mid_val)
	else:
		print('Dimension mismatch between train/test set, splitting train set')

		start_idx = int(0.9*num_train)

		New_X_train = X_train[:start_idx]
		X_test = X_train[start_idx:]

		New_Y_train = Y_train[:start_idx]
		Y_test = Y_train[start_idx:]

		X_train = New_X_train
		Y_train = New_Y_train

		num_test 		= num_train - start_idx
		total_dim_test 	= total_dim

		num_train = start_idx
else:
	# data spliting
	print('No test set given, splitting train set')

	start_idx = int(0.9*num_train)

	New_X_train = X_train[:start_idx]
	X_test = X_train[start_idx:]

	New_Y_train = Y_train[:start_idx]
	Y_test = Y_train[start_idx:]

	X_train = New_X_train
	Y_train = New_Y_train

	num_test 		= num_train - start_idx
	total_dim_test 	= total_dim

	num_train = start_idx

# print size
print('Training size:', np.shape(X_train) )
print('Testing size:', np.shape(X_test) )

print('Average Num Non Zero Train:', np.mean(X_train.getnnz(axis=1)))
print('Max Num Non Zero Train:', np.amax(X_train.getnnz(axis=1)))
print('Min Num Non Zero Train:', np.amin(X_train.getnnz(axis=1)))

print('Average Sparsity Train:', np.mean(X_train.getnnz(axis=1) / total_dim))
print('Max Sparsity Train:', np.amax(X_train.getnnz(axis=1) / total_dim))
print('Min Sparsity Train:', np.amin(X_train.getnnz(axis=1) / total_dim))

# flatten input label
Y_train = Y_train.flatten()
if num_test > 0:
	Y_test = Y_test.flatten()

# normalize data
print("Normalizing data...")
sklearn.preprocessing.normalize(X_train, 'l2', axis=1, copy=False)
if num_test > 0:
	sklearn.preprocessing.normalize(X_test, 'l2', axis=1, copy=False)
print()

# fix a seed
np.random.seed(0)

#=================== Define Function Pointer =====================

# FuncF_Eval: 	evaluate objective function F
# GradComEval: 	evaluate gradient of component function f_i
# FullGradEval: evaluate full gradient
# ProxEval: 	evaluate proximal operator of G
# FuncG_Eval: 	evaluate objective function G

#### loss 1: l(s,t) = 1 - tanh(gamma ts)
if prog_option["LossFunction"] == '1':
	FuncF_Eval 		= func_val_bin_class_loss_1
	GradEval 		= grad_eval_bin_class_loss_1
	GradDiffEval 	= grad_diff_eval_bin_class_loss_1
	OMEGA = 1
	#### The Lipschitz constant of f'
	L = 8*(1+np.sqrt(3))*(2+np.sqrt(3))/(3 + np.sqrt(3))**3*OMEGA**2

# #### loss 2: l(s,t) = (1 - 1 / (1 + exp(-ts)))^2
elif prog_option["LossFunction"] == '2':
	FuncF_Eval 		= func_val_bin_class_loss_2
	GradEval 		= grad_eval_bin_class_loss_2
	GradDiffEval 	= grad_diff_eval_bin_class_loss_2
	#### The Lipschitz constant of f'
	L = 0.15405

# #### loss 3: l(s,t) = log(1+exp(-ts)) - log(1 + exp(-ts - gamma))
if prog_option["LossFunction"] == '3':
	FuncF_Eval 		= func_val_bin_class_loss_3
	GradEval 		= grad_eval_bin_class_loss_3
	GradDiffEval 	= grad_diff_eval_bin_class_loss_3
	#### The Lipschitz constant of f'
	L = 0.1 ## Exact value: 0.092372

# common function pointers
ProxEval = prox_l1_norm
FuncG_Eval = func_val_l1_norm
Acc_Eval = accuracy
# decide whether to perform accuracy evaluation
if num_test > 0 and total_dim_test == total_dim:
	isAccEval = 1
else:
	isAccEval = 0

#=========== Set learning rate for other algorithms ==============

if batch_size == 1:
	# ProxSVRG
	eta_prox_svrg = 5 / (3*L * num_train) # Make learning rate a bit bigger than the theory.
	prox_svrg_inner_batch = batch_size
	max_inner_prox_svrg = num_train

	# ProxSGD
	eta_prox_sgd = 0.01 # initial learning rate
	eta_prime_prox_sgd = 0.5#0.5
	prox_sgd_batch_size = batch_size

else:
	# ProxSPDB
	eta_prox_spdb = 1 / (2*L)
	prox_spdb_inner_batch_size = int(round(np.sqrt(num_train)))   # this is b
	max_inner_prox_spdb = num_train // prox_spdb_inner_batch_size # this is m

	# ProxSVRG
	eta_prox_svrg = 1.0 / (3*L)
	prox_svrg_inner_batch = int(round(num_train**(2.0/3.0)))   # this is b
	max_inner_prox_svrg = num_train // prox_svrg_inner_batch   # this is m.

	# ProxSGD
	eta_prox_sgd = 0.5 # initial learning rate
	eta_prime_prox_sgd = 0.5#0.5
	prox_sgd_batch_size = batch_size

# ProxGD
eta_prox_gd = 1.0/L

# common param
lamb = 1.0 / num_train#0.1/np.sqrt(num_train)
eta_comp = 0.5
max_num_epoch = max_num_epochs

#===================== Parameter Selection =======================

## param selection
c = 0.01
r = 100.0
q = 2 + c + (1/r)
# omega = (1.0/c) + 4.0*(1.0 + r) / (q**2)
omega = (q**2 + 8) / (q**2)
rho = (1/batch_size) * ((num_train  - batch_size) / (num_train - 1.0)) * omega

# param config for ProxSARAH
# note: we set gamma = gamma_const, we can choose a certain mini-batch size to achive better convergence
gamma_const = np.array([0.5, 0.95, 0.99, 0.95, 0.99])

# calculate max inner iteration
# if total_dim > 1e6 or num_train > 1e6:
# 	C_const = np.ones(5)
# else:
# 	C_const = 2.0 / (3.0 * L**2 * gamma_const**2 )
# C_const = 2.0 / (3.0 * L**2 * gamma_const**2 )
C_const = (q**2) / ((q**2 + 8) * L**2 * gamma_const**2 )

C_const = np.maximum(np.minimum(C_const,4), 0.5)

max_inner_prox_sarah = np.zeros(5)
prox_sarah_inner_batch = np.zeros(5)

# b = m = O(sqrt(n)), gamma = 0.95
prox_sarah_inner_batch[1] = int(num_train**(0.5)) / C_const[1] 		# b = sqrt(n)/C
max_inner_prox_sarah[1] = num_train / prox_sarah_inner_batch[1]     				# m = sqrt(n)

# b = m = O(sqrt(n)), gamma = 0.99
prox_sarah_inner_batch[2] = int(num_train**(0.5)) / C_const[2]
max_inner_prox_sarah[2] = num_train / prox_sarah_inner_batch[2]

# b = m = O(n^(1/3)), gamma = 0.95
max_inner_prox_sarah[3] = int(num_train**(1.0/3.0))
prox_sarah_inner_batch[3] = int(num_train**(1.0/3.0)) / C_const[3]

# b = m = O(n^(1/3)), gamma = 0.99
max_inner_prox_sarah[4] = int(num_train**(1.0/3.0))
prox_sarah_inner_batch[4] = int(num_train**(1.0/3.0)) / C_const[4]

# single sample, m = n
max_inner_prox_sarah[0] = num_train 	# m = n
prox_sarah_inner_batch[0] = 1  		 	# b = 1

# convert to integer values
max_inner_prox_sarah = max_inner_prox_sarah.astype(int)
prox_sarah_inner_batch = prox_sarah_inner_batch.astype(int)

# calculate step sizes
eta_prox_sarah = 2.0 / (q + gamma_const * L)
eta_prox_sarah[0] = 2 * np.sqrt(omega * max_inner_prox_sarah[0]) / (q *np.sqrt(omega * max_inner_prox_sarah[0]) + 1)
gamma_prox_sarah = gamma_const # For mini-batch
gamma_prox_sarah[0] = 1/ (L * np.sqrt(omega* max_inner_prox_sarah[0])) # For single sample

# ProxSARAH Adaptive
gamma_m = 0.99
eta_prox_sarah_adaptive = 1.0 / (L * gamma_m)
max_inner_prox_sarah_adaptive = np.zeros(3)
prox_sarah_adaptive_inner_batch = np.zeros(3)

max_inner_prox_sarah_adaptive[0] = num_train
prox_sarah_adaptive_inner_batch[0] = 1

max_inner_prox_sarah_adaptive[1] = int(num_train**(0.5))
prox_sarah_adaptive_inner_batch[1] = int(num_train**(0.5))

max_inner_prox_sarah_adaptive[2] = int(num_train**(1.0/3.0))
prox_sarah_adaptive_inner_batch[2] = int(num_train**(1.0/3.0))

max_inner_prox_sarah_adaptive = max_inner_prox_sarah_adaptive.astype(int)
prox_sarah_adaptive_inner_batch = prox_sarah_adaptive_inner_batch.astype(int)

#================== Generate an initial point ====================

# initial point
# w0 = np.ones(total_dim)
# w0 = (1/np.sqrt(total_dim))*w0
w0 = np.zeros(total_dim)

# Define the bias vector
bias = np.zeros(num_train)

#=================================================================
#=====================  Training Process  ========================
#=================================================================

# record start time
start_train = time.time()

# ProxSARAH single sample
if (alg_list["ProxSARAH"] and prox_sarah_option['1']):
	print('----------------------------------------------------')
	start_prox_sarah1 = time.time()

	w_prox_sarah1, hist_NumGrad_prox_sarah1, hist_NumEpoch_prox_sarah1, \
	hist_TrainLoss_prox_sarah1, hist_GradNorm_prox_sarah1, hist_MinGradNorm_prox_sarah1, \
	hist_TrainAcc_prox_sarah1, hist_TestAcc_prox_sarah1 = prox_sarah(num_train, total_dim, X_train,\
			Y_train, X_test, Y_test, bias, eta_prox_sarah[0], eta_comp, max_num_epoch, max_inner_prox_sarah[0], \
			w0,	gamma_prox_sarah[0], lamb, num_train, prox_sarah_inner_batch[0], GradEval, \
			GradDiffEval, FuncF_Eval, ProxEval, FuncG_Eval, Acc_Eval, isAccEval, verbose, log_enable)

	elapsed_prox_sarah1 = time.time() - start_prox_sarah1
	print("\nTraining time (ProxSARAH single sample): {:^8.2f} seconds\n".format(elapsed_prox_sarah1))

# ProxSARAH-v1
if (alg_list["ProxSARAH"] and prox_sarah_option['2']):
	print('----------------------------------------------------')
	start_prox_sarah2 = time.time()

	w_prox_sarah2, hist_NumGrad_prox_sarah2, hist_NumEpoch_prox_sarah2, \
	hist_TrainLoss_prox_sarah2, hist_GradNorm_prox_sarah2, hist_MinGradNorm_prox_sarah2, \
	hist_TrainAcc_prox_sarah2, hist_TestAcc_prox_sarah2 = prox_sarah(num_train, total_dim,\
			X_train, Y_train, X_test, Y_test, bias, eta_prox_sarah[1], eta_comp, max_num_epoch,\
			max_inner_prox_sarah[1], w0, gamma_prox_sarah[1], lamb, num_train, prox_sarah_inner_batch[1],\
			GradEval, GradDiffEval, FuncF_Eval, ProxEval, FuncG_Eval, Acc_Eval, isAccEval, verbose, log_enable)

	elapsed_prox_sarah2 = time.time() - start_prox_sarah2
	print("\nTraining time (ProxSARAH-v1): {:^8.2f} seconds\n".format(elapsed_prox_sarah2))

# ProxSARAH-v2
if (alg_list["ProxSARAH"] and prox_sarah_option['3']):
	print('----------------------------------------------------')
	start_prox_sarah3 = time.time()

	w_prox_sarah3, hist_NumGrad_prox_sarah3, hist_NumEpoch_prox_sarah3, \
	hist_TrainLoss_prox_sarah3, hist_GradNorm_prox_sarah3, hist_MinGradNorm_prox_sarah3, \
	hist_TrainAcc_prox_sarah3, hist_TestAcc_prox_sarah3= prox_sarah(num_train, total_dim,\
			X_train, Y_train, X_test, Y_test, bias,	eta_prox_sarah[2], eta_comp, max_num_epoch,\
			max_inner_prox_sarah[2], w0, gamma_prox_sarah[2], lamb, num_train, prox_sarah_inner_batch[2],\
			GradEval, GradDiffEval, FuncF_Eval, ProxEval, FuncG_Eval, Acc_Eval, isAccEval, verbose, log_enable)

	elapsed_prox_sarah3 = time.time() - start_prox_sarah3
	print("\nTraining time (ProxSARAH-v2): {:^8.2f} seconds\n".format(elapsed_prox_sarah3))

# ProxSARAH-v3
if (alg_list["ProxSARAH"] and prox_sarah_option['4']):
	print('----------------------------------------------------')
	start_prox_sarah4 = time.time()

	w_prox_sarah4, hist_NumGrad_prox_sarah4, hist_NumEpoch_prox_sarah4, \
	hist_TrainLoss_prox_sarah4, hist_GradNorm_prox_sarah4, hist_MinGradNorm_prox_sarah4, \
	hist_TrainAcc_prox_sarah4, hist_TestAcc_prox_sarah4 = prox_sarah(num_train, total_dim,\
			X_train, Y_train, X_test, Y_test, bias,	eta_prox_sarah[3], eta_comp, max_num_epoch,\
			max_inner_prox_sarah[3], w0, gamma_prox_sarah[3], lamb, num_train, prox_sarah_inner_batch[3],\
			GradEval, GradDiffEval, FuncF_Eval, ProxEval, FuncG_Eval, Acc_Eval, isAccEval, verbose, log_enable)

	elapsed_prox_sarah4 = time.time() - start_prox_sarah4
	print("\nTraining time (ProxSARAH-v3): {:^8.2f} seconds\n".format(elapsed_prox_sarah4))

# ProxSARAH-v4
if (alg_list["ProxSARAH"] and prox_sarah_option['5']):
	print('----------------------------------------------------')
	start_prox_sarah5 = time.time()

	w_prox_sarah5, hist_NumGrad_prox_sarah5, hist_NumEpoch_prox_sarah5, \
	hist_TrainLoss_prox_sarah5, hist_GradNorm_prox_sarah5, hist_MinGradNorm_prox_sarah5, \
	hist_TrainAcc_prox_sarah5, hist_TestAcc_prox_sarah5	= prox_sarah(num_train, total_dim,\
			X_train, Y_train, X_test, Y_test, bias,	eta_prox_sarah[4], eta_comp, max_num_epoch,\
			max_inner_prox_sarah[4], w0, gamma_prox_sarah[4], lamb, num_train, prox_sarah_inner_batch[4],\
			GradEval, GradDiffEval, FuncF_Eval, ProxEval, FuncG_Eval, Acc_Eval, isAccEval, verbose, log_enable)

	elapsed_prox_sarah5 = time.time() - start_prox_sarah5
	print("\nTraining time (ProxSARAH-v4): {:^8.2f} seconds\n".format(elapsed_prox_sarah5))

# ProxSARAH-A-v1
if (alg_list["ProxSARAHAdaptive"] and prox_sarah_adaptive_option['1']):
	print('----------------------------------------------------')
	start_prox_sarah_adaptive1 = time.time()

	w_prox_sarah_adaptive1, hist_NumGrad_prox_sarah_adaptive1, hist_NumEpoch_prox_sarah_adaptive1, \
	hist_TrainLoss_prox_sarah_adaptive1, hist_GradNorm_prox_sarah_adaptive1, hist_MinGradNorm_prox_sarah_adaptive1, \
	hist_TrainAcc_prox_sarah_adaptive1, hist_TestAcc_prox_sarah_adaptive1 = prox_sarah_adaptive(num_train, total_dim,\
			X_train, Y_train, X_test, Y_test, bias,	eta_prox_sarah_adaptive, eta_comp, max_num_epoch,\
			max_inner_prox_sarah_adaptive[0], w0, L, gamma_m, lamb, num_train, prox_sarah_adaptive_inner_batch[0],\
			GradEval, GradDiffEval, FuncF_Eval, ProxEval, FuncG_Eval, Acc_Eval, isAccEval, verbose, log_enable)

	elapsed_prox_sarah_adaptive1 = time.time() - start_prox_sarah_adaptive1
	print("\nTraining time (ProxSARAH-A-v1): {:^8.2f} seconds\n".format(elapsed_prox_sarah_adaptive1))

# ProxSARAH-A-v2
if (alg_list["ProxSARAHAdaptive"] and prox_sarah_adaptive_option['2']):
	print('----------------------------------------------------')
	start_prox_sarah_adaptive2 = time.time()

	w_prox_sarah_adaptive2, hist_NumGrad_prox_sarah_adaptive2, hist_NumEpoch_prox_sarah_adaptive2, \
	hist_TrainLoss_prox_sarah_adaptive2, hist_GradNorm_prox_sarah_adaptive2, hist_MinGradNorm_prox_sarah_adaptive2, \
	hist_TrainAcc_prox_sarah_adaptive2, hist_TestAcc_prox_sarah_adaptive2 = prox_sarah_adaptive(num_train, total_dim,\
			X_train, Y_train, X_test, Y_test, bias, eta_prox_sarah_adaptive, eta_comp, max_num_epoch,\
			max_inner_prox_sarah_adaptive[1], w0, L, gamma_m, lamb, num_train, prox_sarah_adaptive_inner_batch[1],\
			GradEval, GradDiffEval, FuncF_Eval, ProxEval, FuncG_Eval, Acc_Eval, isAccEval, verbose, log_enable)

	elapsed_prox_sarah_adaptive2 = time.time() - start_prox_sarah_adaptive2
	print("\nTraining time (ProxSARAH-A-v2): {:^8.2f} seconds\n".format(elapsed_prox_sarah_adaptive2))

# ProxSARAH-A-v3
if (alg_list["ProxSARAHAdaptive"] and prox_sarah_adaptive_option['3']):
	print('----------------------------------------------------')
	start_prox_sarah_adaptive3 = time.time()

	w_prox_sarah_adaptive3, hist_NumGrad_prox_sarah_adaptive3, hist_NumEpoch_prox_sarah_adaptive3, \
	hist_TrainLoss_prox_sarah_adaptive3, hist_GradNorm_prox_sarah_adaptive3, hist_MinGradNorm_prox_sarah_adaptive3, \
	hist_TrainAcc_prox_sarah_adaptive3, hist_TestAcc_prox_sarah_adaptive3 = prox_sarah_adaptive(num_train, total_dim,\
			X_train, Y_train, X_test, Y_test, bias, eta_prox_sarah_adaptive, eta_comp, max_num_epoch,\
			max_inner_prox_sarah_adaptive[2], w0, L, gamma_m, lamb, num_train, prox_sarah_adaptive_inner_batch[2],\
			GradEval, GradDiffEval, FuncF_Eval, ProxEval, FuncG_Eval, Acc_Eval, isAccEval, verbose, log_enable)

	elapsed_prox_sarah_adaptive3 = time.time() - start_prox_sarah_adaptive3
	print("\nTraining time (ProxSARAH-A-v3): {:^8.2f} seconds\n".format(elapsed_prox_sarah_adaptive3))

# ProxSpiderBoost 
if (alg_list["ProxSpiderBoost"]):
	print('----------------------------------------------------')
	start_prox_spdb = time.time()

	w_prox_spdb, hist_NumGrad_prox_spdb, hist_NumEpoch_prox_spdb, hist_TrainLoss_prox_spdb, \
	hist_GradNorm_prox_spdb, hist_MinGradNorm_prox_spdb, hist_TrainAcc_prox_spdb, hist_TestAcc_prox_spdb \
			= prox_spbd(num_train, total_dim, X_train, Y_train, X_test, Y_test, bias, eta_prox_spdb, eta_comp, \
			max_num_epoch, max_inner_prox_spdb, w0, lamb, num_train,prox_spdb_inner_batch_size,\
			GradEval, GradDiffEval, FuncF_Eval, ProxEval, FuncG_Eval, Acc_Eval, isAccEval, verbose, log_enable)

	elapsed_prox_spdb = time.time() - start_prox_spdb
	print("\nTraining time (ProxSpiderBoost): {:^8.2f} seconds\n".format(elapsed_prox_spdb))

# ProxSVRG 
if (alg_list["ProxSVRG"]):
	print('----------------------------------------------------')
	start_prox_svrg = time.time()

	w_prox_svrg, hist_NumGrad_prox_svrg, hist_NumEpoch_prox_svrg, hist_TrainLoss_prox_svrg, \
	hist_GradNorm_prox_svrg, hist_MinGradNorm_prox_svrg, hist_TrainAcc_prox_svrg, hist_TestAcc_prox_svrg \
			= prox_svrg(num_train, total_dim, X_train, Y_train, X_test, Y_test, bias, eta_prox_svrg,\
			eta_comp, max_num_epoch, max_inner_prox_svrg, w0, lamb, prox_svrg_inner_batch, GradEval, \
			GradDiffEval, FuncF_Eval, ProxEval, FuncG_Eval, Acc_Eval, isAccEval, verbose, log_enable)

	elapsed_prox_svrg = time.time() - start_prox_svrg
	print("\nTraining time (ProxSVRG): {:^8.2f} seconds\n".format(elapsed_prox_svrg))

# ProxSGD 
if (alg_list["ProxSGD"]):
	print('----------------------------------------------------')
	start_prox_sgd = time.time()

	w_prox_sgd, hist_NumGrad_prox_sgd, hist_NumEpoch_prox_sgd, hist_TrainLoss_prox_sgd, \
	hist_GradNorm_prox_sgd, hist_MinGradNorm_prox_sgd, hist_TrainAcc_prox_sgd, hist_TestAcc_prox_sgd \
			= prox_sgd(num_train, total_dim, X_train, Y_train, X_test, Y_test, bias, eta_prox_sgd,\
			eta_prime_prox_sgd, eta_comp, max_num_epoch, w0, lamb, prox_sgd_batch_size, GradEval, \
			FuncF_Eval, ProxEval, FuncG_Eval, Acc_Eval, isAccEval, verbose, log_enable)
	
	elapsed_prox_sgd = time.time() - start_prox_sgd
	print("\nTraining time (ProxSGD): {:^8.2f} seconds\n".format(elapsed_prox_sgd))

# ProxGD
if (alg_list["ProxGD"]):
	print('----------------------------------------------------')
	start_prox_gd = time.time()

	w_prox_gd, hist_NumGrad_prox_gd, hist_NumEpoch_prox_gd, hist_TrainLoss_prox_gd, \
	hist_GradNorm_prox_gd, hist_MinGradNorm_prox_gd, hist_TrainAcc_prox_gd, hist_TestAcc_prox_gd \
			= prox_gd(num_train, total_dim, X_train, Y_train, X_test, Y_test, bias, eta_prox_gd, \
			eta_comp, max_num_epoch, w0, lamb, GradEval, FuncF_Eval, ProxEval, \
			FuncG_Eval, Acc_Eval, isAccEval, verbose, log_enable)

	elapsed_prox_gd = time.time() - start_prox_gd
	print("\nTraining time (ProxGD): {:^8.2f} seconds\n".format(elapsed_prox_gd))

# record time elapsed
elapsed_train = time.time() - start_train
print("Total training time: {:^8.2f} seconds\n".format(elapsed_train))

#=================================================================
#=======================  Plot Process  ==========================
#=================================================================

examplename = 'Binary Classification'

if plot_option:

	#=================================================================
	# Plot Training Loss

	fig1 = plt.figure()

	if (alg_list["ProxSARAH"] and prox_sarah_option['1']):
		plt.plot(np.array(hist_NumEpoch_prox_sarah1), hist_TrainLoss_prox_sarah1, 'b-', label = 'ProxSARAH single sample')	

	if (alg_list["ProxSARAH"] and prox_sarah_option['2']):
		plt.plot(np.array(hist_NumEpoch_prox_sarah2), hist_TrainLoss_prox_sarah2, 'C0-', label = 'ProxSARAH b=sqrt(n), gamma = 0.95')

	if (alg_list["ProxSARAH"] and prox_sarah_option['3']):
		plt.plot(np.array(hist_NumEpoch_prox_sarah3), hist_TrainLoss_prox_sarah3, 'C1-', label = 'ProxSARAH b=sqrt(n), gamma = 0.99')

	if (alg_list["ProxSARAH"] and prox_sarah_option['4']):
		plt.plot(np.array(hist_NumEpoch_prox_sarah4), hist_TrainLoss_prox_sarah4, 'C2-', label = 'ProxSARAH b=n^(1/3), gamma = 0.95')

	if (alg_list["ProxSARAH"] and prox_sarah_option['5']):
		plt.plot(np.array(hist_NumEpoch_prox_sarah5), hist_TrainLoss_prox_sarah5, 'C3-', label = 'ProxSARAH b=n^(1/3), gamma = 0.99')	
	
	if (alg_list["ProxSARAHAdaptive"] and prox_sarah_adaptive_option['1']):
		plt.plot(np.array(hist_NumEpoch_prox_sarah_adaptive1), hist_TrainLoss_prox_sarah_adaptive1, 'C4-', label = 'ProxSARAH Adaptive single sample')
	
	if (alg_list["ProxSARAHAdaptive"] and prox_sarah_adaptive_option['2']):
		plt.plot(np.array(hist_NumEpoch_prox_sarah_adaptive2), hist_TrainLoss_prox_sarah_adaptive2, 'C5-', label = 'ProxSARAH Adaptive b=sqrt(n)')

	if (alg_list["ProxSARAHAdaptive"] and prox_sarah_adaptive_option['3']):
		plt.plot(np.array(hist_NumEpoch_prox_sarah_adaptive3), hist_TrainLoss_prox_sarah_adaptive3, 'C6-', label = 'ProxSARAH Adaptive b=n^(1/3)')

	if (alg_list["ProxSpiderBoost"]):
		plt.plot(np.array(hist_NumEpoch_prox_spdb), hist_TrainLoss_prox_spdb, 'C9-', label = 'ProxSpiderBoost')
	
	if (alg_list["ProxSVRG"]):
		plt.plot(np.array(hist_NumEpoch_prox_svrg), hist_TrainLoss_prox_svrg, 'C8--', label = 'ProxSVRG')	
	
	if (alg_list["ProxSGD"]):
		plt.plot(np.array(hist_NumEpoch_prox_sgd), hist_TrainLoss_prox_sgd, 'g-.', label = 'ProxSGD')	

	if (alg_list["ProxGD"]):
		plt.plot(np.array(hist_NumEpoch_prox_gd), hist_TrainLoss_prox_gd, 'C7-.', label = 'ProxGD')

	fig1.suptitle("Training Loss - " + examplename + ' - ' + data_name)
	plt.xlabel("Number of Effective Passes")
	plt.ylabel("Training Loss")
	plt.legend()
	plt.show()

	#=================================================================
	# Plot norm gradient mapping square

	fig2 = plt.figure()

	if (alg_list["ProxSARAH"] and prox_sarah_option['1']):
		plt.semilogy(np.array(hist_NumEpoch_prox_sarah1), hist_GradNorm_prox_sarah1, 'b-', label = 'ProxSARAH single sample')

	if (alg_list["ProxSARAH"] and prox_sarah_option['2']):
		plt.semilogy(np.array(hist_NumEpoch_prox_sarah2), hist_GradNorm_prox_sarah2, 'C0-', label = 'ProxSARAH b=sqrt(n), gamma = 0.95')

	if (alg_list["ProxSARAH"] and prox_sarah_option['3']):
		plt.semilogy(np.array(hist_NumEpoch_prox_sarah3), hist_GradNorm_prox_sarah3, 'C1-', label = 'ProxSARAH b=sqrt(n), gamma = 0.99')

	if (alg_list["ProxSARAH"] and prox_sarah_option['4']):
		plt.semilogy(np.array(hist_NumEpoch_prox_sarah4), hist_GradNorm_prox_sarah4, 'C2-', label = 'ProxSARAH b=n^(1/3), gamma = 0.95')

	if (alg_list["ProxSARAH"] and prox_sarah_option['5']):
		plt.semilogy(np.array(hist_NumEpoch_prox_sarah5), hist_GradNorm_prox_sarah5, 'C3-', label = 'ProxSARAH b=n^(1/3), gamma = 0.99')	

	if (alg_list["ProxSARAHAdaptive"] and prox_sarah_adaptive_option['1']):
		plt.semilogy(np.array(hist_NumEpoch_prox_sarah_adaptive1), hist_GradNorm_prox_sarah_adaptive1, 'C4-', label = 'ProxSARAH Adaptive single sample')
	
	if (alg_list["ProxSARAHAdaptive"] and prox_sarah_adaptive_option['2']):
		plt.semilogy(np.array(hist_NumEpoch_prox_sarah_adaptive2), hist_GradNorm_prox_sarah_adaptive2, 'C5-', label = 'ProxSARAH Adaptive b=sqrt(n)')

	if (alg_list["ProxSARAHAdaptive"] and prox_sarah_adaptive_option['3']):
		plt.semilogy(np.array(hist_NumEpoch_prox_sarah_adaptive3), hist_GradNorm_prox_sarah_adaptive3, 'C6-', label = 'ProxSARAH Adaptive b=n^(1/3)')

	if (alg_list["ProxSpiderBoost"]):
		plt.semilogy(np.array(hist_NumEpoch_prox_spdb), hist_GradNorm_prox_spdb, 'C9-', label = 'ProxSpiderBoost')
	
	if (alg_list["ProxSVRG"]):
		plt.semilogy(np.array(hist_NumEpoch_prox_svrg), hist_GradNorm_prox_svrg, 'C8--', label = 'ProxSVRG')

	if (alg_list["ProxSGD"]):
		plt.semilogy(np.array(hist_NumEpoch_prox_sgd), hist_GradNorm_prox_sgd, 'g-.', label = 'ProxSGD')

	if (alg_list["ProxGD"]):
		plt.semilogy(np.array(hist_NumEpoch_prox_gd), hist_GradNorm_prox_gd, 'C7-.', label = 'ProxGD')

	fig2.suptitle("Norm Grad Mapping Square - " + examplename + ' - ' +data_name)
	plt.xlabel("Number of Effective Passes")
	plt.ylabel("Norm Grad Mapping Square")
	plt.legend()
	plt.show()

	#=================================================================
	# Plot min norm gradient mapping square

	fig3 = plt.figure()

	if (alg_list["ProxSARAH"] and prox_sarah_option['1']):
		plt.semilogy(np.array(hist_NumEpoch_prox_sarah1), hist_MinGradNorm_prox_sarah1, 'b-', label = 'ProxSARAH single sample')

	if (alg_list["ProxSARAH"] and prox_sarah_option['2']):
		plt.semilogy(np.array(hist_NumEpoch_prox_sarah2), hist_MinGradNorm_prox_sarah2, 'C0-', label = 'ProxSARAH b=sqrt(n), gamma = 0.95')

	if (alg_list["ProxSARAH"] and prox_sarah_option['3']):
		plt.semilogy(np.array(hist_NumEpoch_prox_sarah3), hist_MinGradNorm_prox_sarah3, 'C1-', label = 'ProxSARAH b=sqrt(n), gamma = 0.99')

	if (alg_list["ProxSARAH"] and prox_sarah_option['4']):
		plt.semilogy(np.array(hist_NumEpoch_prox_sarah4), hist_MinGradNorm_prox_sarah4, 'C2-', label = 'ProxSARAH b=n^(1/3), gamma = 0.95')

	if (alg_list["ProxSARAH"] and prox_sarah_option['5']):
		plt.semilogy(np.array(hist_NumEpoch_prox_sarah5), hist_MinGradNorm_prox_sarah5, 'C3-', label = 'ProxSARAH b=n^(1/3), gamma = 0.99')	

	if (alg_list["ProxSARAHAdaptive"] and prox_sarah_adaptive_option['1']):
		plt.semilogy(np.array(hist_NumEpoch_prox_sarah_adaptive1), hist_MinGradNorm_prox_sarah_adaptive1, 'C4-', label = 'ProxSARAH Adaptive single sample')
	
	if (alg_list["ProxSARAHAdaptive"] and prox_sarah_adaptive_option['2']):
		plt.semilogy(np.array(hist_NumEpoch_prox_sarah_adaptive2), hist_MinGradNorm_prox_sarah_adaptive2, 'C5-', label = 'ProxSARAH Adaptive b=sqrt(n)')

	if (alg_list["ProxSARAHAdaptive"] and prox_sarah_adaptive_option['3']):
		plt.semilogy(np.array(hist_NumEpoch_prox_sarah_adaptive3), hist_MinGradNorm_prox_sarah_adaptive3, 'C6-', label = 'ProxSARAH Adaptive b=n^(1/3)')

	if (alg_list["ProxSpiderBoost"]):
		plt.semilogy(np.array(hist_NumEpoch_prox_spdb), hist_MinGradNorm_prox_spdb, 'C9-', label = 'ProxSpiderBoost')
	
	if (alg_list["ProxSVRG"]):
		plt.semilogy(np.array(hist_NumEpoch_prox_svrg), hist_MinGradNorm_prox_svrg, 'C8--', label = 'ProxSVRG')

	if (alg_list["ProxSGD"]):
		plt.semilogy(np.array(hist_NumEpoch_prox_sgd), hist_MinGradNorm_prox_sgd, 'g-.', label = 'ProxSGD')

	if (alg_list["ProxGD"]):
		plt.semilogy(np.array(hist_NumEpoch_prox_gd), hist_MinGradNorm_prox_gd, 'C7-.', label = 'ProxGD')

	fig3.suptitle("Min Norm Grad Mapping Square - " + examplename + ' - ' +data_name)
	plt.xlabel("Number of Effective Passes")
	plt.ylabel("Min Norm Grad Mapping Square")
	plt.legend()
	plt.show()

	if isAccEval:
		#=================================================================
		# Plot Train Accuracy

		fig4 = plt.figure()

		if (alg_list["ProxSARAH"] and prox_sarah_option['1']):
			plt.plot(np.array(hist_NumEpoch_prox_sarah1), hist_TrainAcc_prox_sarah1, 'b-', label = 'ProxSARAH single sample')

		if (alg_list["ProxSARAH"] and prox_sarah_option['2']):
			plt.plot(np.array(hist_NumEpoch_prox_sarah2), hist_TrainAcc_prox_sarah2, 'C0-', label = 'ProxSARAH b=sqrt(n), gamma = 0.95')

		if (alg_list["ProxSARAH"] and prox_sarah_option['3']):
			plt.plot(np.array(hist_NumEpoch_prox_sarah3), hist_TrainAcc_prox_sarah3, 'C1-', label = 'ProxSARAH b=sqrt(n), gamma = 0.99')

		if (alg_list["ProxSARAH"] and prox_sarah_option['4']):
			plt.plot(np.array(hist_NumEpoch_prox_sarah4), hist_TrainAcc_prox_sarah4, 'C2-', label = 'ProxSARAH b=n^(1/3), gamma = 0.95')

		if (alg_list["ProxSARAH"] and prox_sarah_option['5']):
			plt.plot(np.array(hist_NumEpoch_prox_sarah5), hist_TrainAcc_prox_sarah5, 'C3-', label = 'ProxSARAH b=n^(1/3), gamma = 0.99')	

		if (alg_list["ProxSARAHAdaptive"] and prox_sarah_adaptive_option['1']):
			plt.plot(np.array(hist_NumEpoch_prox_sarah_adaptive1), hist_TrainAcc_prox_sarah_adaptive1, 'C4-', label = 'ProxSARAH Adaptive single sample')

		if (alg_list["ProxSARAHAdaptive"] and prox_sarah_adaptive_option['2']):
			plt.plot(np.array(hist_NumEpoch_prox_sarah_adaptive2), hist_TrainAcc_prox_sarah_adaptive2, 'C5-', label = 'ProxSARAH Adaptive b=sqrt(n)')

		if (alg_list["ProxSARAHAdaptive"] and prox_sarah_adaptive_option['3']):
			plt.plot(np.array(hist_NumEpoch_prox_sarah_adaptive3), hist_TrainAcc_prox_sarah_adaptive3, 'C6-', label = 'ProxSARAH Adaptive b=n^(1/3)')

		if (alg_list["ProxSpiderBoost"]):
			plt.plot(np.array(hist_NumEpoch_prox_spdb), hist_TrainAcc_prox_spdb, 'C9-', label = 'ProxSpiderBoost')

		if (alg_list["ProxSVRG"]):
			plt.plot(np.array(hist_NumEpoch_prox_svrg), hist_TrainAcc_prox_svrg, 'C8--', label = 'ProxSVRG')

		if (alg_list["ProxSGD"]):
			plt.plot(np.array(hist_NumEpoch_prox_sgd), hist_TrainAcc_prox_sgd, 'g-.', label = 'ProxSGD')

		if (alg_list["ProxGD"]):
			plt.plot(np.array(hist_NumEpoch_prox_gd), hist_TrainAcc_prox_gd, 'C7-.', label = 'ProxGD')

		fig4.suptitle("Train Accuracy - " + examplename + ' - ' +data_name)
		plt.xlabel("Number of Effective Passes")
		plt.ylabel("Train Accuracy")
		plt.legend()
		plt.show()

		#=================================================================
		# Plot Test Accuracy

		fig5 = plt.figure()

		if (alg_list["ProxSARAH"] and prox_sarah_option['1']):
			plt.plot(np.array(hist_NumEpoch_prox_sarah1), hist_TestAcc_prox_sarah1, 'b-', label = 'ProxSARAH single sample')

		if (alg_list["ProxSARAH"] and prox_sarah_option['2']):
			plt.plot(np.array(hist_NumEpoch_prox_sarah2), hist_TestAcc_prox_sarah2, 'C0-', label = 'ProxSARAH b=sqrt(n), gamma = 0.95')

		if (alg_list["ProxSARAH"] and prox_sarah_option['3']):
			plt.plot(np.array(hist_NumEpoch_prox_sarah3), hist_TestAcc_prox_sarah3, 'C1-', label = 'ProxSARAH b=sqrt(n), gamma = 0.99')

		if (alg_list["ProxSARAH"] and prox_sarah_option['4']):
			plt.plot(np.array(hist_NumEpoch_prox_sarah4), hist_TestAcc_prox_sarah4, 'C2-', label = 'ProxSARAH b=n^(1/3), gamma = 0.95')

		if (alg_list["ProxSARAH"] and prox_sarah_option['5']):
			plt.plot(np.array(hist_NumEpoch_prox_sarah5), hist_TestAcc_prox_sarah5, 'C3-', label = 'ProxSARAH b=n^(1/3), gamma = 0.99')	

		if (alg_list["ProxSARAHAdaptive"] and prox_sarah_adaptive_option['1']):
			plt.plot(np.array(hist_NumEpoch_prox_sarah_adaptive1), hist_TestAcc_prox_sarah_adaptive1, 'C4-', label = 'ProxSARAH Adaptive single sample')

		if (alg_list["ProxSARAHAdaptive"] and prox_sarah_adaptive_option['2']):
			plt.plot(np.array(hist_NumEpoch_prox_sarah_adaptive2), hist_TestAcc_prox_sarah_adaptive2, 'C5-', label = 'ProxSARAH Adaptive b=sqrt(n)')

		if (alg_list["ProxSARAHAdaptive"] and prox_sarah_adaptive_option['3']):
			plt.plot(np.array(hist_NumEpoch_prox_sarah_adaptive3), hist_TestAcc_prox_sarah_adaptive3, 'C6-', label = 'ProxSARAH Adaptive b=n^(1/3)')

		if (alg_list["ProxSpiderBoost"]):
			plt.plot(np.array(hist_NumEpoch_prox_spdb), hist_TestAcc_prox_spdb, 'C9-', label = 'ProxSpiderBoost')

		if (alg_list["ProxSVRG"]):
			plt.plot(np.array(hist_NumEpoch_prox_svrg), hist_TestAcc_prox_svrg, 'C8--', label = 'ProxSVRG')

		if (alg_list["ProxSGD"]):
			plt.plot(np.array(hist_NumEpoch_prox_sgd), hist_TestAcc_prox_sgd, 'g-.', label = 'ProxSGD')

		if (alg_list["ProxGD"]):
			plt.plot(np.array(hist_NumEpoch_prox_gd), hist_TestAcc_prox_gd, 'C7-.', label = 'ProxGD')

		fig5.suptitle("Test Accuracy - " + examplename + ' - ' +data_name)
		plt.xlabel("Number of Effective Passes")
		plt.ylabel("Test Accuracy")
		plt.legend()
		plt.show()

#=================================================================
