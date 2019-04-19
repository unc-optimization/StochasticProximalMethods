"""! @package method_ProxGD

Implementation of ProxGD algorithm presented in

Y. Nesterov. Introductory lectures on convex optimization: A basic course, volume 87 of Applied Optimization. Kluwer Academic Publishers, 2004.

The algorithm is used to solve the nonconvex composite problem
    
\f $ F(w) = E_{\zeta_i} [f(w,\zeta_i)] + g(w) \f $

which covers the finite sum as a special case

\f $ F(w) = \frac{1}{n} \sum_{i=1}^n (f_i(w)) + g(w). \f $

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

#library import
import numpy as np

#===============================================================================================================================
# ProxGD

def prox_gd(n, d, X_train, Y_train, X_test, Y_test, bias, eta, eta_comp, max_num_epoch, w0, lamb, GradEval, FuncF_Eval, \
			ProxEval, FuncG_Eval, Acc_Eval, isAccEval, verbose = 0, is_fun_eval = 1):

	"""! ProxGD algorithm

	Parameters
	----------
	@param n : sample size
	@param d : number of features
	@param X_train : train data
	@param Y_train : train label
	@param X_test : test data
	@param Y_test : test label
	@param bias : bias vector
	@param eta : learning rate
	@param eta_comp : common learning rate used for gradient mapping squared norm comparsion between algorithms
	@param max_num_epoch : the minimum number of epochs to run before termination
	@param w0 : initial point
	@param lamb : penalty parameter of the non-smooth objective
	@param GradEval : function pointer for gradient of f
	@param GradDiffEval : function pointer for difference of gradient nablaf(w') - nablaf(w)
	@param FuncF_Eval : function pointer to compute objective value of f(w)
	@param ProxEval : function pointer to compute proximal operator of g(w)
	@param FuncG_Eval : function pointer to compute objective value of g(w)
	@param Acc_Eval : function pointer to compute accuracy
	@param isAccEval : flag whether to compute accuracy
	@param verbose : specify verbosity level

			0 : silence

			1 : print iteration info

	@param is_fun_eval : flag whether to compute and log data

	Returns
	-------
	@retval w : solution
	@retval hist_TrainLoss : train loss history
	@retval hist_NumGrad : number of gradient evaluations history
	@retval hist_GradNorm : squared norm of gradient mapping history
	@retval hist_MinGradNorm : minimum squared norm of gradient mapping history
	@retval hist_NumEpoch : history of epochs at which data were recorded
	@retval hist_TrainAcc : train accuracy history
	@retval hist_TestAcc : test accuracy history
	"""

	# initialize history list
	hist_TrainLoss 		= []
	hist_NumGrad 		= []
	hist_NumEpoch 		= []
	hist_GradNorm 		= []
	hist_TrainAcc 		= []
	hist_TestAcc 		= []
	hist_MinGradNorm 	= []

	# initialize stats variables
	min_norm_grad_map 	= 1.0e6

	# Count number of component gradient evaluations
	num_grad = 0
	num_epoch = 0

	# count total number of iterations
	total_iter = 0

	# get length of test data
	num_test = len(Y_test)

	# get average number of non zero elements in training data
	nnz_Xtrain = np.mean(X_train.getnnz(axis=1))
	if isAccEval:
		nnz_Xtest = np.mean(X_test.getnnz(axis=1))

	# print initial message
	if verbose:
		print('Start ProxGD...', '\neta = ', eta)
	
	# Assign initial value
	w = w0

	# calculate full gradient
	v_cur, XYw = GradEval(n, d, n, X_train, Y_train, bias, w, nnz_Xtrain)
	
	# log data
	if is_fun_eval:

		# calculate gradient mapping for stats report
		grad_map = (1/(eta_comp)) *(w - ProxEval(w - eta_comp*v_cur, lamb*eta_comp))
		norm_grad_map = np.dot(grad_map.T, grad_map)

		# update mins
		if norm_grad_map < min_norm_grad_map:
			min_norm_grad_map = norm_grad_map
	
		# Get Training Loss
		train_loss = FuncF_Eval(n, XYw) + lamb * FuncG_Eval(w)

		# calculate test accuracy
		if isAccEval:
			train_accuracy = 1/float(n) * np.sum( 1*(XYw > 0) )
			test_accuracy = Acc_Eval(num_test, d, X_test, Y_test, bias, w, nnz_Xtest)
	
		# print info
		if verbose:
			print("Epoch:", num_epoch, "\nTraining Loss: ", train_loss)
			if isAccEval:
				print("Train Accuracy = {:.3f}".format(train_accuracy), "\nTest Accuracy = {:.3f}".format(test_accuracy))
			print("||Gradient mapping||^2: ", norm_grad_map, "\nmin ||Gradient Mapping||^2: ", min_norm_grad_map, "\n")

		# update history if requires
		hist_TrainLoss.append(train_loss)
		if isAccEval:
			hist_TrainAcc.append(train_accuracy)
			hist_TestAcc.append(test_accuracy)
		hist_GradNorm.append(np.asscalar(norm_grad_map))
		hist_MinGradNorm.append(min_norm_grad_map)
		hist_NumGrad.append(num_grad)
		hist_NumEpoch.append(num_epoch)

	# Main loop
	while num_epoch < max_num_epoch:

		# Algorithm update
		w = ProxEval( w - eta*v_cur, lamb*eta )

		# calculate full gradient 
		v_cur, XYw = GradEval(n, d, n, X_train, Y_train, bias, w, nnz_Xtrain)
		
		# Increase number of component gradient (1 full gradient = n component gradient)
		num_grad += n
		num_epoch += 1
		
		# log data
		if is_fun_eval:
			# calculate gradient norm square and gradient mapping for stats report
			norm_grad = np.dot(v_cur.T,v_cur)
			grad_map = (1/(eta_comp)) *(w - ProxEval(w - eta_comp*v_cur, lamb*eta_comp))
			norm_grad_map = np.dot(grad_map.T, grad_map)

			# update mins
			if norm_grad_map < min_norm_grad_map:
				min_norm_grad_map = norm_grad_map

			# Get Training Loss
			train_loss = FuncF_Eval(n, XYw) + lamb * FuncG_Eval(w)

			# calculate test accuracy
			if isAccEval:
				train_accuracy = 1/float(n) * np.sum( 1*(XYw > 0) )
				test_accuracy = Acc_Eval(num_test, d, X_test, Y_test, bias, w, nnz_Xtest)
		
			# print info
			if verbose:
				print("Epoch:", num_epoch, "\nTraining Loss: ", train_loss)
				if isAccEval:
					print("Train Accuracy = {:.3f}".format(train_accuracy), "\nTest Accuracy = {:.3f}".format(test_accuracy))
				print("||Gradient mapping||^2: ", norm_grad_map, "\nmin ||Gradient Mapping||^2: ", min_norm_grad_map, "\n")

			# update history if requires
			hist_TrainLoss.append(train_loss)
			if isAccEval:
				hist_TrainAcc.append(train_accuracy)
				hist_TestAcc.append(test_accuracy)
			hist_GradNorm.append(np.asscalar(norm_grad_map))
			hist_MinGradNorm.append(min_norm_grad_map)
			hist_NumGrad.append(num_grad)
			hist_NumEpoch.append(num_epoch)
	# Main loop ends
	
	return w, hist_NumGrad, hist_NumEpoch, hist_TrainLoss, hist_GradNorm, hist_MinGradNorm, hist_TrainAcc, hist_TestAcc

#===============================================================================================================================
