# import the necessary packages
import argparse
import numpy as np
 
def argParser():
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

	ap.add_argument("-lo", "--loss", required=False,
		help="1: use loss function 1 in binary classification example\
			  2: use loss function 2 in binary classification example\
			  3: use loss function 3 in binary classification example\
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

	ap.add_argument("-aso", "--AdaptiveSarahOption", required=False,
		help="Select minibatch and step sizes:\n \
				0: all\n\
				1: b = 1\n\
				1: b = m = O(sqrt(n))\n\
				2: b = m = O(n^(1/3))\n\
				")

	ap.add_argument("-id", "--identification", required=False,
		help="unique ID number")

	# read arguments
	args = ap.parse_args()

	# create a dictionary to stor program parameters
	prog_option = {}

	# check whether to plot
	prog_option["PlotOption"] = 0
	if args.plot:
		prog_option["PlotOption"] = int(args.plot)

	prog_option["MaxNumEpoch"] = 15
	if args.numepoch:
		prog_option["MaxNumEpoch"] = int(args.numepoch)
		
	prog_option["Verbose"] = 1
	if args.verbose:
		prog_option["Verbose"] = int(args.verbose)

	prog_option["LogEnable"] = 1
	if args.log:
		prog_option["LogEnable"] = args.log

	prog_option["LossFunction"] = '1'
	if args.loss:
		prog_option["LossFunction"] = args.loss

	# get starting id
	if args.identification:
		prog_option["ProgID"] = args.identification
	else:
		prog_option["ProgID"] = ''	

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
					"ProxSARAHAdaptive" : 	0,
					"ProxSpiderBoost"	:	0,
					"ProxSVRG"			:	0,
					"ProxSGD"			:	0,
					"ProxGD"			:	0
					}

	if args.algorithms:
		if '1' in args.algorithms:
		    print ('Enable Prox SARAH')
		    alg_list["ProxSARAH"] = 1
		
		if '2' in args.algorithms:
		    print ('Prox SARAH Adaptive')
		    alg_list["ProxSARAHAdaptive"] = 1
		
		if '3' in args.algorithms:
		    print ('Prox Spider Boost')
		    alg_list["ProxSpiderBoost"] = 1
		
		if '4' in args.algorithms:
		    print ('Prox SVRG')
		    alg_list["ProxSVRG"] = 1
		
		if '5' in args.algorithms:
		    print ('Prox SGD')
		    alg_list["ProxSGD"] = 1
		
		if '6' in args.algorithms:
		    print ('Prox GD')
		    alg_list["ProxGD"] = 1

		if '0' in args.algorithms:
			print('Enable all algorithms')
			alg_list["ProxSARAH"] 			= 1
			alg_list["ProxSARAHAdaptive"] 	= 1
			alg_list["ProxSpiderBoost"] 	= 1
			alg_list["ProxSVRG"] 			= 1
			alg_list["ProxSGD"] 			= 1
			alg_list["ProxGD"] 				= 1

	else:
		print('No algorithm selected, running Prox SARAH')
		alg_list["ProxSARAH"] 			= 1

	# prox_sarah_option = np.zeros(5)
	prox_sarah_option = {		# 0: disable, 1: enable
					'1' :	0, 	# single sample
					'2' : 	0, 	# b = m = O(sqrt(n)), gamma = 0.75
					'3' :	0, 	# b = m = O(sqrt(n)), gamma = 0.95
					'4' :	0, 	# b = m = O(n^(1/3)), gamma = 0.75
					'5' :	0, 	# b = m = O(n^(1/3)), gamma = 0.95
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

	prox_sarah_adaptive_option = {	# 0: disable, 1: enable
						'1' :	0, 	# single sample
						'2' : 	0, 	# b = m = O(sqrt(n))
						'3' :	0, 	# b = m = O(n^(1/3))
				}
	if args.AdaptiveSarahOption:
		if '1' in args.AdaptiveSarahOption:
			prox_sarah_adaptive_option['1'] = 1
		if '2' in args.AdaptiveSarahOption:
			prox_sarah_adaptive_option['2'] = 1
		if '3' in args.AdaptiveSarahOption:
			prox_sarah_adaptive_option['3'] = 1
	else:
		prox_sarah_adaptive_option['1'] = 1

	return data_name, prog_option, alg_list, prox_sarah_option, prox_sarah_adaptive_option