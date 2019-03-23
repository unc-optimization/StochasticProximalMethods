"""!@package import_data

Useful function to read different dataset.

Supported file: .csv or LIBSVM datasets.
"""

############################
# written by Lam M. Nguyen
# Import Data
############################

import numpy as np
from sklearn.datasets import *
import numpy as np
from joblib import Memory

from pathlib import Path
from csv import reader
import sys
import os

# Important: change these paths to folder containing datasets according to your setup.
# data_path = '/home/nhanph/dataset/'

# check if dataset path exists
if not os.path.exists(data_path):
	sys.exit("\033[91m {}\033[00m" .format("Error: Dataset not found!!!"))

# set a location to store cache file
# default: same folder as dataset folder
mem = Memory(data_path + "mycache")

@mem.cache
def import_data(data_option):
	"""! Import dataset

	Depending on the name of dataset, this function will return the corresponding input.
	
	@Note: for libsvm dataset, only the prefix are needed.
		Ex: the data contains data.tr and data.t, you only need to set data_option = 'data'

	Parameters
	----------
	@param data_option : name of the dataset
	    
	Returns
	-------
	@retval X_train : train data
	@retval New_Y_train : train label
	retval X_test : test data
	@retval New_Y_test : test label
	"""
	if (data_option == 'mnist'): 

		### ==================== MNIST DATA ===========================

		mnist_path = data_path + 'MNIST_data/'

		if sys.version_info[0] == 2:
			from urllib import urlretrieve
		else:
			from urllib.request import urlretrieve

		def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
			print("Downloading %s" % filename)
			urlretrieve(source + filename, mnist_path + filename)
		
		import gzip
		
		def load_mnist_images(filename):
			if not os.path.exists(mnist_path + filename):
				download(filename)
			with gzip.open(mnist_path + filename, 'rb') as f:
				data = np.frombuffer(f.read(), np.uint8, offset=16)
			data = data.reshape(-1, 784)
			return data

		def load_mnist_labels(filename):
			if not os.path.exists(mnist_path + filename):
				download(filename)
			with gzip.open(mnist_path + filename, 'rb') as f:
				data = np.frombuffer(f.read(), np.uint8, offset=8)
			return data
			
		# Get Training and Test Data
		X_train = load_mnist_images('train-images-idx3-ubyte.gz')
		Y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
		X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
		Y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

		# scale data
		X_train = X_train / 255
		X_test = X_test / 255

		# Get sizes of training data
		num_train = len(Y_train)

		# Get size of test data
		num_test = len(Y_test)

		num_classes = 10

		# categorize training data
		Temp_Y_train = np.zeros((num_train, num_classes))

		# set corresponding position to 1
		for i in range(num_train):
			Temp_Y_train[i,Y_train[i]] = 1

		New_Y_train = Temp_Y_train

		# categorize test data
		Temp_Y_test = np.zeros((num_test, num_classes))
		
		for i in range(num_test):
			Temp_Y_test[i,Y_test[i]] = 1

		New_Y_test = Temp_Y_test

		### ==========================================================		

	elif (data_option == 'optdigits'):

		### ==================== OPTDIGITS DATA ===========================	

		source_path = data_path
		
		# Train Data
		filename_train = 'optdigits.tra'
		train_data = load_csv(source_path + filename_train)
		for i in range(len(train_data[0])):
			str_column_to_float(train_data, i)

		train_data = np.array(train_data)	
		X_train = train_data[:,0:64]
		Y_train = train_data[:,64]

		# Test Data
		filename_test = 'optdigits.tes'
		test_data = load_csv(source_path + filename_test)
		for i in range(len(test_data[0])):
			str_column_to_float(test_data, i)

		test_data = np.array(test_data)	
		X_test = test_data[:,0:64]
		Y_test = test_data[:,64]

		# Normalize data
		max_val = np.max(X_train)
		min_val = np.min(X_train)
		X_train = (X_train - min_val)/(max_val - min_val)
		X_test = (X_test - min_val)/(max_val - min_val)

		# Get sizes of training data
		num_train, num_feature = np.shape(X_train)

		# Get size of test data
		num_test = len(Y_test)

		# Convert label to -1 (0,1,2,3,4) and +1 (5,6,7,8,9)
		New_Y_train = np.zeros(num_train)
		New_Y_test = np.zeros(num_test)

		for i in range(0,num_train):
			if (Y_train[i] != 1):
				New_Y_train[i] = -1
			else:
				New_Y_train[i] = 1
				
		for i in range(0,num_test):
			if (Y_test[i] != 1):
				New_Y_test[i] = -1
			else:
				New_Y_test[i] = 1

		### ==========================================================	

	elif (data_option == 'news20'): 

		### ==================== NEWS20 DATA ===========================

		train = fetch_20newsgroups_vectorized(subset='train')
		test = fetch_20newsgroups_vectorized(subset='test')

		X_train_sparse = train.data
		X_test_sparse = test.data

		# Convert sparse matrices
		X_train = X_train_sparse.toarray()
		X_test = X_test_sparse.toarray()

		Y_train = train.target
		Y_test = test.target

		# Normalize data
		max_val = np.max(X_train)
		min_val = np.min(X_train)
		X_train = (X_train - min_val)/(max_val - min_val)
		X_test = (X_test - min_val)/(max_val - min_val)

		# Get sizes of training data
		num_train, num_feature = np.shape(X_train)

		# Get size of test data
		num_test = len(Y_test)

		# Convert label to -1 and +1
		New_Y_train = np.zeros(num_train)
		New_Y_test = np.zeros(num_test)

		for i in range(0,num_train):
			if (Y_train[i] != 10):
				New_Y_train[i] = -1
			else:
				New_Y_train[i] = 1
				
		for i in range(0,num_test):
			if (Y_test[i] != 10):
				New_Y_test[i] = -1
			else:
				New_Y_test[i] = 1

		### ==========================================================	

	elif (data_option == 'covtype'):

		### ==================== covtype DATA ===========================	

		source_path = data_path
		
		# X Data
		filename_x = 'covtype_x_data.csv'
		x_data = load_csv(source_path + filename_x)
		for i in range(len(x_data[0])):
			str_column_to_float(x_data, i)

		x_data = np.array(x_data)
		len_x_data, _ = np.shape(x_data)
		sep_len = len_x_data*7//10	
		X_train = x_data[:sep_len]
		X_test = x_data[sep_len:]

		# Y Data
		filename_y = 'covtype_y_data.csv'
		y_data = load_csv(source_path + filename_y)
		for i in range(len(y_data[0])):
			str_column_to_float(y_data, i)

		y_data = np.array(y_data)
		Y_train = y_data[:sep_len]
		Y_test = y_data[sep_len:]

		# Normalize data
		max_val = np.max(X_train)
		min_val = np.min(X_train)
		X_train = (X_train - min_val)/(max_val - min_val)
		X_test = (X_test - min_val)/(max_val - min_val)

		# Get sizes of training data
		num_train, num_feature = np.shape(X_train)

		# Get size of test data
		num_test = len(Y_test)

		# Convert label to -1 (1) and +1 (2)
		New_Y_train = Y_train*2 - 3
		New_Y_test = Y_test*2 - 3

		### ==========================================================	

	elif (data_option == 'ijcnn1'):

		### ==================== ijcnn1 DATA ===========================	

		source_path = data_path
		
		# Train Data
		filename_x_train = 'ijcnn1_x_train.csv'
		train_x_data = load_csv(source_path + filename_x_train)
		for i in range(len(train_x_data[0])):
			str_column_to_float(train_x_data, i)

		train_x_data = np.array(train_x_data)	
		X_train = train_x_data

		filename_y_train = 'ijcnn1_y_train.csv'
		train_y_data = load_csv(source_path + filename_y_train)
		for i in range(len(train_y_data[0])):
			str_column_to_float(train_y_data, i)

		train_y_data = np.array(train_y_data)	
		Y_train = train_y_data

		# Test Data
		filename_x_test = 'ijcnn1_x_test.csv'
		test_x_data = load_csv(source_path + filename_x_test)
		for i in range(len(test_x_data[0])):
			str_column_to_float(test_x_data, i)

		test_x_data = np.array(test_x_data)	
		X_test = test_x_data

		filename_y_test = 'ijcnn1_y_test.csv'
		test_y_data = load_csv(source_path + filename_y_test)
		for i in range(len(test_y_data[0])):
			str_column_to_float(test_y_data, i)

		test_y_data = np.array(test_y_data)	
		Y_test = test_y_data

		# Normalize data
		max_val = np.max(X_train)
		min_val = np.min(X_train)
		X_train = (X_train - min_val)/(max_val - min_val)
		X_test = (X_test - min_val)/(max_val - min_val)

		# Get sizes of training data
		num_train, num_feature = np.shape(X_train)

		# Get size of test data
		num_test = len(Y_test)

		New_Y_train = Y_train
		New_Y_test = Y_test

		### ==========================================================	

	elif (data_option == 'phishing'):

		### ==================== phishing DATA ===========================	

		source_path = data_path
		
		# X Data
		filename_x = 'phishing_x_data.csv'
		x_data = load_csv(source_path + filename_x)
		for i in range(len(x_data[0])):
			str_column_to_float(x_data, i)

		x_data = np.array(x_data)
		len_x_data, _ = np.shape(x_data)
		sep_len = len_x_data*7//10	
		X_train = x_data[:sep_len]
		X_test = x_data[sep_len:]

		# Y Data
		filename_y = 'phishing_y_data.csv'
		y_data = load_csv(source_path + filename_y)
		for i in range(len(y_data[0])):
			str_column_to_float(y_data, i)

		y_data = np.array(y_data)
		Y_train = y_data[:sep_len]
		Y_test = y_data[sep_len:]

		# Normalize data
		max_val = np.max(X_train)
		min_val = np.min(X_train)
		X_train = (X_train - min_val)/(max_val - min_val)
		X_test = (X_test - min_val)/(max_val - min_val)

		# Get sizes of training data
		num_train, num_feature = np.shape(X_train)

		# Get size of test data
		num_test = len(Y_test)

		# Convert label to -1 (0) and +1 (1)
		New_Y_train = Y_train*2 - 1
		New_Y_test = Y_test*2 - 1

		### ==========================================================	

	elif (data_option == 'w8a'):

		### ==================== w8a DATA ===========================	

		source_path = data_path
		
		# Train Data
		filename_x_train = 'w8a_x_train.csv'
		train_x_data = load_csv(source_path + filename_x_train)
		for i in range(len(train_x_data[0])):
			str_column_to_float(train_x_data, i)

		train_x_data = np.array(train_x_data)	
		X_train = train_x_data

		filename_y_train = 'w8a_y_train.csv'
		train_y_data = load_csv(source_path + filename_y_train)
		for i in range(len(train_y_data[0])):
			str_column_to_float(train_y_data, i)

		train_y_data = np.array(train_y_data)	
		Y_train = train_y_data

		# Test Data
		filename_x_test = 'w8a_x_test.csv'
		test_x_data = load_csv(source_path + filename_x_test)
		for i in range(len(test_x_data[0])):
			str_column_to_float(test_x_data, i)

		test_x_data = np.array(test_x_data)	
		X_test = test_x_data

		filename_y_test = 'w8a_y_test.csv'
		test_y_data = load_csv(source_path + filename_y_test)
		for i in range(len(test_y_data[0])):
			str_column_to_float(test_y_data, i)

		test_y_data = np.array(test_y_data)	
		Y_test = test_y_data

		# Normalize data
		max_val = np.max(X_train)
		min_val = np.min(X_train)
		X_train = (X_train - min_val)/(max_val - min_val)
		X_test = (X_test - min_val)/(max_val - min_val)

		# Get sizes of training data
		num_train, num_feature = np.shape(X_train)

		# Get size of test data
		num_test = len(Y_test)

		New_Y_train = Y_train
		New_Y_test = Y_test

		### ==========================================================	
	
	elif (data_option == 'blog'):

		### ==================== blog DATA ===========================	

		source_path = data_path

		# Train Data
		filename_x_train = 'blogData_train.csv'
		train_x_data = load_csv(source_path + filename_x_train)
		for i in range(len(train_x_data[0])):
			str_column_to_float(train_x_data, i)

		train_x_data = np.array(train_x_data)	

		X_train = train_x_data

		# Normalize data
		max_val = np.max(X_train)
		min_val = np.min(X_train)
		X_train = (X_train - min_val)/(max_val - min_val)
		New_Y_train = []
		X_test = []
		New_Y_test = []

	elif (data_option == 'YearPredictionMSD'):

		### ==================== YearPredictionMSD DATA ===========================	

		source_path = data_path

		# Train Data
		filename_x_train = 'YearPredictionMSD.txt'
		train_x_data = load_csv(source_path + filename_x_train)
		for i in range(len(train_x_data[0])):
			str_column_to_float(train_x_data, i)

		train_x_data = np.array(train_x_data)	

		X_train = train_x_data

		# Normalize data
		max_val = np.max(X_train)
		min_val = np.min(X_train)
		X_train = (X_train - min_val)/(max_val - min_val)
		New_Y_train = []
		X_test = []
		New_Y_test = []

	else:
		### ==================== libsvm DATA ===========================	

		source_path = data_path

		train_ext = ['.tr', '.train','']

		for text in train_ext:
			# Train Data
			filename_train = data_option + text
			trainFile = Path(source_path + filename_train)
			if trainFile.is_file():
				X_train, New_Y_train = load_svmlight_file(source_path + filename_train)
				break
			else:
				X_train = []
				New_Y_train = []

		test_ext = ['.t', '.test']

		for text in test_ext:
			filename_test = data_option + text
			testFile = Path(source_path + filename_test)
			if testFile.is_file():
				X_test, New_Y_test = load_svmlight_file(source_path + filename_test)
				break
			else:
				X_test = []
				New_Y_test = []		

	return X_train, New_Y_train, X_test, New_Y_test

def load_csv(filename):
	"""! Load a CSV file

	Parameters
	----------
	@param filename : name of the file
	    
	Returns
	-------
	@retval dataset : list containing input data
	"""
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

def str_column_to_float(dataset, column):
	"""! Convert string column to float

	Parameters
	----------
	@param dataset : input data
	@param column : column index
	    
	"""
	for row in dataset:
		row[column] = float(row[column].strip())

def split_dataset(data_option, percentage):
	"""! Split a dataset and save as train and test set in the same folder

	Parameters
	----------
	@param data_option : name of the dataset
	@percentage	: portion of train set
	    
	"""
	# todo: add content