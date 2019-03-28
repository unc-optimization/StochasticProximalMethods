"""! @package argParser

Parse argument from user command line input.

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

# external library
import argparse
import numpy as np
 
def argParser():
	"""! Argument parser

	This function reads input argument from command line and returns corresponding program options.
	
	@Note: some options are only needed in certain examples.

	For more information, see

		python non_neg_pca_example.py -h

		python binary_classification_example.py -h
		
	Parameters
	----------
	@param none
	    
	Returns
	-------
	@retval data_name : dataset name
	@retval prog_option : list of different program options such as batch size, number of total epochs
	@retval alg_list : list of selected algorithms
	@retval prox_sarah_option : list of selected ProxSARAH variants
	"""
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--plot", required=False,
		help="enable/disable plotting")

	ap.add_argument("-v", "--verbose", required=False,
		help="	0: silent run\
				1: print essential info\
				2: print iteration details\
			  ")

	ap.add_argument("-ne", "--numepoch", required=False,
		help="input maximum number of epochs")

	ap.add_argument("-l", "--log", required=False,
		help="1: log the data\
			  0: no data logging\
			  ")

	ap.add_argument("-b", "--batch", required=False,
		help="mini batch size used in ProxSGD")

	ap.add_argument("-d", "--data", required=False,
		help="input the name/prefix of the dataset. \
			Supported extension: \
			- train data: tr,train\
			- test data: t,test\
			Ex: data.tr & data.t => -d data	\
			Ex: ndata & ndata.t => -d ndata	\
				")

	ap.add_argument("-a", "--algorithms", required=False,
		help="Select algorithm:\n \
				0: all\n\
				1: ProxSARAH\n\
				2: ProxSARAHAdaptive\n\
				3: ProxSpiderBoost\n\
				4: ProxSVRG\n\
				5: ProxSGD\n\
				6: ProxGD\n\
				")

	ap.add_argument("-so", "--SarahOption", required=False,
		help="Select minibatch and step sizes:\n \
				0: all\n\
				1: b = 1\n\
				2: b = m = O(sqrt(n)), gamma = 0.75\n\
				3: b = m = O(sqrt(n)), gamma = 0.95\n\
				4: b = m = O(n^(1/3)), gamma = 0.75\n\
				5: b = m = O(n^(1/3)), gamma = 0.95\n\
				")

	# read arguments
	args = ap.parse_args()

	# create a dictionary to stor program parameters
	prog_option = {}

	# check whether to plot
	prog_option["PlotOption"] = 0
	if args.plot:
		prog_option["PlotOption"] = int(args.plot)

	# max number of epochs to run
	prog_option["MaxNumEpoch"] = 15
	if args.numepoch:
		prog_option["MaxNumEpoch"] = int(args.numepoch)
		
	# verbosity level
	prog_option["Verbose"] = 1
	if args.verbose:
		prog_option["Verbose"] = int(args.verbose)

	# whether to evaluate iteration info
	prog_option["LogEnable"] = 1
	if args.log:
		prog_option["LogEnable"] = args.log

	# get mini batch size
	if args.batch:
		prog_option["BatchSize"] = int(args.batch)
	else:
		prog_option["BatchSize"] = 1

	# get dataset name
	if args.data:
		data_name = args.data
		print('Dataset selected:',data_name)
		
	else:
		data_name = 'phishing'
		print('No dataset selected, running default dataset:', data_name)

	# select algorithms
	alg_list = {	# 0: disable, 1: enable
					"ProxSARAH"			:	0,
					"ProxSpiderBoost"	:	0,
					"ProxSVRG"			:	0,
					"ProxSGD"			:	0,
					}

	if args.algorithms:
		if '1' in args.algorithms:
		    print ('Prox SARAH')
		    alg_list["ProxSARAH"] = 1
		
		if '2' in args.algorithms:
		    print ('Prox Spider Boost')
		    alg_list["ProxSpiderBoost"] = 1
		
		if '3' in args.algorithms:
		    print ('Prox SVRG')
		    alg_list["ProxSVRG"] = 1
		
		if '4' in args.algorithms:
		    print ('Prox SGD')
		    alg_list["ProxSGD"] = 1

		if '0' in args.algorithms:
			print('Select all algorithms')
			alg_list["ProxSARAH"] 			= 1
			alg_list["ProxSpiderBoost"] 	= 1
			alg_list["ProxSVRG"] 			= 1
			alg_list["ProxSGD"] 			= 1

	else:
		print('No algorithm selected, running Prox SARAH')
		alg_list["ProxSARAH"] 			= 1

	# prox_sarah_option = np.zeros(5)
	prox_sarah_option = {		# 0: disable, 1: enable
					'1' :	0, 	# single sample
					'2' : 	0, 	# b = m = O(sqrt(n)), gamma = 0.95
					'3' :	0, 	# b = m = O(sqrt(n)), gamma = 0.99
					'4' :	0, 	# b = m = O(n^(1/3)), gamma = 0.95
					'5' :	0, 	# b = m = O(n^(1/3)), gamma = 0.99
					}
	if args.SarahOption:
		if '1' in args.SarahOption:
			prox_sarah_option['1'] = 1
		if '2' in args.SarahOption:
			prox_sarah_option['2'] = 1
		if '3' in args.SarahOption:
			prox_sarah_option['3'] = 1
		if '4' in args.SarahOption:
			prox_sarah_option['4'] = 1	
		if '5' in args.SarahOption:
			prox_sarah_option['5'] = 1	
		if '0' in args.SarahOption:
			prox_sarah_option['1'] = 1
			prox_sarah_option['2'] = 1
			prox_sarah_option['3'] = 1
			prox_sarah_option['4'] = 1
			prox_sarah_option['5'] = 1
	else:
		prox_sarah_option['1'] = 1

	return data_name, prog_option, alg_list, prox_sarah_option
