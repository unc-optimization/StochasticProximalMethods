"""@package neural_net_example2

This package implements the feedforward neural network example with a fully connected neural network of size 784x800x10 for dataset "mnist" and "fashion_mnist".

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
import keras
from keras import backend as K
import sklearn as skl
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from models import *
from argParser import *
from load_data import *

# import methods
from method_Prox_SARAH import *
from method_Prox_SPDB import *
from method_Prox_SVRG import *
from method_Prox_SGD import *

## USAGE
# python neural_net_example2s.py -d mnist -a 1234 -so 2345 -b 250 -ne 15

#==================================================================================================================

# read arguments
data_name, prog_option, alg_list, prox_sarah_option, prox_sarah_adaptive_option = argParser()

# get program parameters:
plot_option     = prog_option["PlotOption"]
batch_size      = prog_option["BatchSize"]
max_num_epochs  = prog_option["MaxNumEpoch"]
verbose         = prog_option["Verbose"]
log_enable      = prog_option["LogEnable"]
prog_id         = prog_option["ProgID"]

#==================================================================================================================
## import data

# load data
x_train, y_train, x_test, y_test = load_data(data_name)

# print size
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# normalize data
with tf.Session() as sess:
    x_train = sess.run(tf.linalg.l2_normalize(x_train, axis= (1,2)))
    x_test = sess.run(tf.linalg.l2_normalize(x_test, axis= (1,2)))

#==================================================================================================================
## model

if K.image_data_format() == 'channels_first':
    img_rows, img_cols, num_channel = x_train.shape[2], x_train.shape[3], x_train.shape[1]
else:
    img_rows, img_cols, num_channel = x_train.shape[1], x_train.shape[2], x_train.shape[3]

x = tf.placeholder(tf.float32, (None, img_rows, img_cols, num_channel))
y = tf.placeholder(tf.int32, (None, y_train.shape[1]))

#==================================================================================================================
## operation definition

input_size = img_rows * img_cols * num_channel
output_size = y_train.shape[1]
# load models
logits, logits_dup, w_list, w_list_dup = model2(x, input_size, output_size)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.stop_gradient(y))
loss_operation = tf.reduce_mean(cross_entropy)

cross_entropy_dup = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_dup, labels=tf.stop_gradient(y))
loss_operation_dup = tf.reduce_mean(cross_entropy_dup)

#==================================================================================================================
## model evaluation

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

#==================================================================================================================
## set up learning rates, batch sizes

num_examples = len(x_train)

# Lipschitz constant
if data_name == 'mnist':
    L = 5.0 # mnist\
elif data_name == 'fashion_mnist':
    L = 11.0 # fashion_mnist

if batch_size == 1:
    # ProxSVRG
    eta_prox_svrg = 0.005
    prox_svrg_inner_batch = batch_size
    max_inner_prox_svrg = num_examples

    # ProxSGD
    eta_prox_sgd = 0.1 # initial learning rate
    eta_prime_prox_sgd = 0.05
    prox_sgd_batch_size = batch_size
else:
    # ProxSPDB
    if data_name == 'mnist':
        eta_prox_spdb = 0.16 # mnist
    elif data_name == 'fashion_mnist':
        eta_prox_spdb = 0.05 # fashion_mnist

    prox_spdb_inner_batch_size = int(round(np.sqrt(num_examples)))
    max_inner_prox_spdb = num_examples // prox_spdb_inner_batch_size

    # ProxSVRG
    if data_name == 'mnist':
        eta_prox_svrg = 0.2 # mnist
    elif data_name == 'fashion_mnist':
        eta_prox_svrg = 0.09 # fashion_mnist
    
    prox_svrg_inner_batch = int(round(num_examples**(2.0/3.0)))
    max_inner_prox_svrg = num_examples // prox_svrg_inner_batch

    # ProxSGD
    eta_prox_sgd = 0.1
    eta_prime_prox_sgd = 0.5#0.05
    prox_sgd_batch_size = batch_size

#==================================================================================================================
## param selection

# common param
lamb = 1.0 / (num_examples + 0.0)
eta_comp = 0.5
max_num_epoch = max_num_epochs

# params for ProxSARAH
c = 0.01
r = 100.0
q = 2 + c + (1/r)
omega = (q**2 + 8) / (q**2)
rho = (1/batch_size) * ((num_examples  - batch_size) / (num_examples - 1.0)) * omega

# param config for ProxSARAH
# note: we set gamma = gamma_const, we can choose a certain mini-batch size to achive better convergence
gamma_const = np.array([0.5, 0.95, 0.99, 0.95, 0.99])

# calculate max inner iteration
C_const = np.ones(5)

max_inner_prox_sarah = np.zeros(5)
prox_sarah_inner_batch = np.zeros(5)

# b = m = O(sqrt(n)), gamma = 0.95
prox_sarah_inner_batch[1] = int(num_examples**(0.5)) / C_const[1]
max_inner_prox_sarah[1] = num_examples / prox_sarah_inner_batch[1]s

# b = m = O(sqrt(n)), gamma = 0.99
prox_sarah_inner_batch[2] = int(num_examples**(0.5)) / C_const[2]
max_inner_prox_sarah[2] = num_examples / prox_sarah_inner_batch[2]

# b = m = O(n^(1/3)), gamma = 0.95
max_inner_prox_sarah[3] = int(num_examples**(1.0/3.0))
prox_sarah_inner_batch[3] = int(num_examples**(1.0/3.0)) / C_const[3]

# b = m = O(n^(1/3)), gamma = 0.99
max_inner_prox_sarah[4] = int(num_examples**(1.0/3.0))
prox_sarah_inner_batch[4] = int(num_examples**(1.0/3.0)) / C_const[4]

# single sample, m = n
max_inner_prox_sarah[0] = num_examples
prox_sarah_inner_batch[0] = 1

# convert to integer values
max_inner_prox_sarah = max_inner_prox_sarah.astype(int)
prox_sarah_inner_batch = prox_sarah_inner_batch.astype(int)

# calculate step sizes
eta_prox_sarah = 2.0 / (q + gamma_const * L)
eta_prox_sarah[0] = 2 * np.sqrt(omega * max_inner_prox_sarah[0]) / (q *np.sqrt(omega * max_inner_prox_sarah[0]) + 1)
gamma_prox_sarah = gamma_const
gamma_prox_sarah[0] = 1.0/ (L * np.sqrt(omega* max_inner_prox_sarah[0]))

#=================================================================
#=====================  Training Process  ========================
#=================================================================

# record start time
start_train = time.time()

# ProxSARAH single sample
if (alg_list["ProxSARAH"] and prox_sarah_option['1']):
    w_prox_sarah1, hist_NumGrad_prox_sarah1, hist_NumEpoch_prox_sarah1, \
    hist_TrainLoss_prox_sarah1, hist_GradNorm_prox_sarah1, hist_MinGradNorm_prox_sarah1, \
    hist_TrainAcc_prox_sarah1, hist_TestAcc_prox_sarah1 = Prox_SARAH(x, y, x_train, y_train, x_test, y_test, eta_prox_sarah[0], eta_comp, lamb, gamma_prox_sarah[0],\
        num_examples, prox_sarah_inner_batch[0], max_num_epoch, max_inner_prox_sarah[0], \
        w_list, loss_operation, accuracy_operation, verbose, log_enable)

# ProxSARAH-v1
if (alg_list["ProxSARAH"] and prox_sarah_option['2']):
    w_prox_sarah2, hist_NumGrad_prox_sarah2, hist_NumEpoch_prox_sarah2, \
    hist_TrainLoss_prox_sarah2, hist_GradNorm_prox_sarah2, hist_MinGradNorm_prox_sarah2, \
    hist_TrainAcc_prox_sarah2, hist_TestAcc_prox_sarah2 = Prox_SARAH(x, y, x_train, y_train, x_test, y_test, eta_prox_sarah[1], eta_comp, lamb, gamma_prox_sarah[1],\
        num_examples, prox_sarah_inner_batch[1], max_num_epoch, max_inner_prox_sarah[1], \
        w_list, loss_operation, accuracy_operation, verbose, log_enable)

# ProxSARAH-v2
if (alg_list["ProxSARAH"] and prox_sarah_option['3']):
    w_prox_sarah3, hist_NumGrad_prox_sarah3, hist_NumEpoch_prox_sarah3, \
    hist_TrainLoss_prox_sarah3, hist_GradNorm_prox_sarah3, hist_MinGradNorm_prox_sarah3, \
    hist_TrainAcc_prox_sarah3, hist_TestAcc_prox_sarah3 = Prox_SARAH(x, y, x_train, y_train, x_test, y_test, eta_prox_sarah[2], eta_comp, lamb, gamma_prox_sarah[2],\
        num_examples, prox_sarah_inner_batch[2], max_num_epoch, max_inner_prox_sarah[2], \
        w_list, loss_operation, accuracy_operation, verbose, log_enable)

# ProxSARAH-v3
if (alg_list["ProxSARAH"] and prox_sarah_option['4']):
    w_prox_sarah4, hist_NumGrad_prox_sarah4, hist_NumEpoch_prox_sarah4, \
    hist_TrainLoss_prox_sarah4, hist_GradNorm_prox_sarah4, hist_MinGradNorm_prox_sarah4, \
    hist_TrainAcc_prox_sarah4, hist_TestAcc_prox_sarah4 = Prox_SARAH(x, y, x_train, y_train, x_test, y_test, eta_prox_sarah[3], eta_comp, lamb, gamma_prox_sarah[3],\
        num_examples, prox_sarah_inner_batch[3], max_num_epoch, max_inner_prox_sarah[3], \
        w_list, loss_operation, accuracy_operation, verbose, log_enable)

# ProxSARAH-v4
if (alg_list["ProxSARAH"] and prox_sarah_option['5']):
    w_prox_sarah5, hist_NumGrad_prox_sarah5, hist_NumEpoch_prox_sarah5, \
    hist_TrainLoss_prox_sarah5, hist_GradNorm_prox_sarah5, hist_MinGradNorm_prox_sarah5, \
    hist_TrainAcc_prox_sarah5, hist_TestAcc_prox_sarah5 = Prox_SARAH(x, y, x_train, y_train, x_test, y_test, eta_prox_sarah[4], eta_comp, lamb, gamma_prox_sarah[4],\
        num_examples, prox_sarah_inner_batch[4], max_num_epoch, max_inner_prox_sarah[4], \
        w_list, loss_operation, accuracy_operation, verbose, log_enable)

# ProxSpiderBoost 
if (alg_list["ProxSpiderBoost"]):
    w_prox_spdb, hist_NumGrad_prox_spdb, hist_NumEpoch_prox_spdb, hist_TrainLoss_prox_spdb, \
    hist_GradNorm_prox_spdb, hist_MinGradNorm_prox_spdb, hist_TrainAcc_prox_spdb, hist_TestAcc_prox_spdb \
        = Prox_SPDBoost(x, y, x_train, y_train, x_test, y_test, eta_prox_spdb, eta_comp, lamb,\
        num_examples, prox_spdb_inner_batch_size, max_num_epoch, max_inner_prox_spdb, \
        w_list, loss_operation, accuracy_operation, verbose, log_enable)

# ProxSVRG 
if (alg_list["ProxSVRG"]):
    w_prox_svrg, hist_NumGrad_prox_svrg, hist_NumEpoch_prox_svrg, hist_TrainLoss_prox_svrg, \
    hist_GradNorm_prox_svrg, hist_MinGradNorm_prox_svrg, hist_TrainAcc_prox_svrg, hist_TestAcc_prox_svrg \
        = Prox_SVRG(x, y, x_train, y_train, x_test,\
        y_test, prox_svrg_inner_batch, eta_prox_svrg, eta_comp, lamb, \
        max_num_epoch, max_inner_prox_svrg, w_list, w_list_dup, loss_operation, \
        loss_operation_dup, accuracy_operation, verbose, log_enable)

# ProxSGD 
if (alg_list["ProxSGD"]):
    w_prox_sgd, hist_NumGrad_prox_sgd, hist_NumEpoch_prox_sgd, hist_TrainLoss_prox_sgd, \
    hist_GradNorm_prox_sgd, hist_MinGradNorm_prox_sgd, hist_TrainAcc_prox_sgd, hist_TestAcc_prox_sgd \
        = Prox_SGD(x, y, x_train, y_train, x_test, y_test, \
        prox_sgd_batch_size, eta_prox_sgd, eta_prime_prox_sgd, eta_comp, lamb, max_num_epoch,\
        w_list, loss_operation, accuracy_operation, verbose, log_enable)

# record time elapsed
elapsed_train = time.time() - start_train
print("Total training time: ", elapsed_train)

#=================================================================
#=======================  Plot Process  ==========================
#=================================================================

examplename = 'Neural Nets'

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

    if (alg_list["ProxSpiderBoost"]):
        plt.plot(np.array(hist_NumEpoch_prox_spdb), hist_TrainLoss_prox_spdb, 'C9-', label = 'ProxSpiderBoost')
    
    if (alg_list["ProxSVRG"]):
        plt.plot(np.array(hist_NumEpoch_prox_svrg), hist_TrainLoss_prox_svrg, 'C8--', label = 'ProxSVRG')   
    
    if (alg_list["ProxSGD"]):
        plt.plot(np.array(hist_NumEpoch_prox_sgd), hist_TrainLoss_prox_sgd, 'g-.', label = 'ProxSGD')   

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

    if (alg_list["ProxSpiderBoost"]):
        plt.semilogy(np.array(hist_NumEpoch_prox_spdb), hist_GradNorm_prox_spdb, 'C9-', label = 'ProxSpiderBoost')
    
    if (alg_list["ProxSVRG"]):
        plt.semilogy(np.array(hist_NumEpoch_prox_svrg), hist_GradNorm_prox_svrg, 'C8--', label = 'ProxSVRG')

    if (alg_list["ProxSGD"]):
        plt.semilogy(np.array(hist_NumEpoch_prox_sgd), hist_GradNorm_prox_sgd, 'g-.', label = 'ProxSGD')

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
    
    if (alg_list["ProxSpiderBoost"]):
        plt.semilogy(np.array(hist_NumEpoch_prox_spdb), hist_MinGradNorm_prox_spdb, 'C9-', label = 'ProxSpiderBoost')
    
    if (alg_list["ProxSVRG"]):
        plt.semilogy(np.array(hist_NumEpoch_prox_svrg), hist_MinGradNorm_prox_svrg, 'C8--', label = 'ProxSVRG')

    if (alg_list["ProxSGD"]):
        plt.semilogy(np.array(hist_NumEpoch_prox_sgd), hist_MinGradNorm_prox_sgd, 'g-.', label = 'ProxSGD')

    fig3.suptitle("Min Norm Grad Mapping Square - " + examplename + ' - ' +data_name)
    plt.xlabel("Number of Effective Passes")
    plt.ylabel("Min Norm Grad Mapping Square")
    plt.legend()
    plt.show()

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

    if (alg_list["ProxSpiderBoost"]):
        plt.plot(np.array(hist_NumEpoch_prox_spdb), hist_TrainAcc_prox_spdb, 'C9-', label = 'ProxSpiderBoost')

    if (alg_list["ProxSVRG"]):
        plt.plot(np.array(hist_NumEpoch_prox_svrg), hist_TrainAcc_prox_svrg, 'C8--', label = 'ProxSVRG')

    if (alg_list["ProxSGD"]):
        plt.plot(np.array(hist_NumEpoch_prox_sgd), hist_TrainAcc_prox_sgd, 'g-.', label = 'ProxSGD')

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

    if (alg_list["ProxSpiderBoost"]):
        plt.plot(np.array(hist_NumEpoch_prox_spdb), hist_TestAcc_prox_spdb, 'C9-', label = 'ProxSpiderBoost')

    if (alg_list["ProxSVRG"]):
        plt.plot(np.array(hist_NumEpoch_prox_svrg), hist_TestAcc_prox_svrg, 'C8--', label = 'ProxSVRG')

    if (alg_list["ProxSGD"]):
        plt.plot(np.array(hist_NumEpoch_prox_sgd), hist_TestAcc_prox_sgd, 'g-.', label = 'ProxSGD')

    fig5.suptitle("Test Accuracy - " + examplename + ' - ' +data_name)
    plt.xlabel("Number of Effective Passes")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.show()

#=================================================================
