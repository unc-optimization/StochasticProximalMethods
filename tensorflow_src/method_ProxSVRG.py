"""! @package method_ProxSVRG

Implementation of ProxSVRG algorithm.

The algorithm is used to solve the nonconvex composite problem
    
\f $ F(w) = E_{\zeta_i} [f(w,\zeta_i)] + g(w) \f $

which covers the finite sum as a special case

\f $ F(w) = \frac{1}{n} \sum_{i=1}^n (f_i(w)) + g(w) \f $

This algorithm is implemented specifically in the case \f$g(w) = \|w\|_1 \f$.

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
import numpy as np
from sklearn.utils import shuffle
import pandas as pd

from utils import *


#==================================================================================================================
# ProxSVRG

def Prox_SVRG(x, y, x_train, y_train, x_test, y_test, inner_batch_size, LR, LR_COMP, LBD, MAX_TOTAL_EPOCH, MAX_INNER_ITERATION, \
                w_list, w_list_dup, loss_operation, loss_operation_dup, accuracy_operation, verbose = 1, log_enable = 1):
    
    """! ProxSVRG algorithm

    Parameters
    ----------
    @param x : placeholder for input data
    @param y : placeholder for input label
    @param x_train : train data
    @param y_train : train label
    @param x_test : test data
    @param y_test : test label
    @param inner_batch_size : batch size used to calculate gradient difference in the inner loop
    @param LR : learning rate
    @param LR_COMP : common learning rate used for gradient mapping squared norm comparsion between algorithms
    @param LBD : penalty parameter of the non-smooth objective
    @param MAX_TOTAL_EPOCH : the minimum number of epochs to run before termination
    @param MAX_INNER_ITERATION : maximum number of inner loop's iterations
    @param w_list : list containing trainable parameters
    @param w_list_dup : a copy of trainable parameters
    @param loss_operation : operation to evaluate loss
    @param loss_operation_dup : a copy of operation to evaluate loss
    @param prog_id : function pointer for difference of gradient nablaf(w') - nablaf(w)
    @param accuracy_operation : operation to evaluate accuracy
    @param verbose : specify verbosity level

            0 : silence

            1 : print iteration info

    @param log_enable : flag whether to compute and log data

    Returns
    -------
    @retval w : solution
    @retval hist_NumGrad : number of gradient evaluations history
    @retval hist_NumEpoch : history of epochs at which data were recorded
    @retval hist_TrainLoss : train loss history    
    @retval hist_GradNorm : squared norm of gradient mapping history
    @retval hist_MinGradNorm : minimum squared norm of gradient mapping history
    @retval hist_TrainAcc : train accuracy history
    @retval hist_TestAcc : test accuracy history
    """

    #=========================================#
    # Setup

    ## create a snapshot vector
    w0_list = w_list_dup

    ## operation for calculating gradient
    grad_list       = tf.gradients(loss_operation, w_list)
    grad_list_w0    = tf.gradients(loss_operation_dup, w0_list)

    ## parameters used in algorithm update
    scale   = tf.placeholder(tf.float32)
    lr      = tf.placeholder(tf.float32)
    lbd     = tf.placeholder(tf.float32)
    gm      = tf.placeholder(tf.float32)

    ## variables for main updates
    v0_list         = [tf.Variable(tf.zeros(list(w.get_shape())), dtype = tf.float32) for w in w0_list]
    v_list          = [tf.Variable(tf.zeros(list(w.get_shape())), dtype = tf.float32) for w in w_list]
    grad_map_list   = [tf.Variable(tf.zeros(list(w.get_shape())), dtype = tf.float32) for w in w_list]

    ## variables to hold norm square of gradient, gradient mapping and l1 norm of w
    norm_v0_sq          = tf.Variable(0.0)
    norm_v_sq           = tf.Variable(0.0)
    norm_grad_map_sq    = tf.Variable(0.0)
    norm_l1_w           = tf.Variable(0.0)

    ## supporting operations in main loop
    # operations used when calculating gradients
    ops_set_v0_to_zero  = []
    ops_add_grad_to_v0  = []
    ops_assign_w0       = []
    ops_svrg_update_v   = []

    # operations used when calculating norm of gradients
    ops_set_norm_v0_to_zero         = []
    ops_set_norm_v_to_zero          = []
    ops_set_norm_grad_map_to_zero   = []
    ops_set_norm_l1_w_to_zero       = []
    ops_calc_norm_v0_sq              = []
    ops_calc_norm_v_sq               = []
    ops_calc_norm_grad_map_sq        = []
    ops_calc_norm_l1_w               = []

    # operations for main algorithm update
    ops_update_w        = []
    ops_update_grad_map = []

    # define operations in main SVRG loop
    for v0, v, w, w0, grad, grad_w0, grad_map in zip(v0_list, v_list, \
        w_list, w0_list, grad_list, grad_list_w0, grad_map_list):

        # Full gradient operations
        ops_set_v0_to_zero.append(v0.assign(v0*0))
        ops_add_grad_to_v0.append(v0.assign_add(scale*grad))
        
        # store w0
        ops_assign_w0.append(w0.assign(w))

        # calculate norm squares and l1 norm
        ops_set_norm_v0_to_zero.append(norm_v0_sq.assign(norm_v0_sq * 0))
        ops_set_norm_v_to_zero.append(norm_v_sq.assign(norm_v_sq * 0))        
        ops_set_norm_grad_map_to_zero.append(norm_grad_map_sq.assign(norm_grad_map_sq * 0))
        ops_set_norm_l1_w_to_zero.append(norm_l1_w.assign(norm_l1_w * 0))

        ops_calc_norm_v0_sq.append(norm_v0_sq.assign_add(tf.reduce_sum(tf.multiply(v0,v0))))
        ops_calc_norm_v_sq.append(norm_v_sq.assign_add(tf.reduce_sum(tf.multiply(v,v))))
        ops_calc_norm_grad_map_sq.append(norm_grad_map_sq.assign_add(tf.reduce_sum(tf.multiply(grad_map, grad_map))))
        ops_calc_norm_l1_w.append(norm_l1_w.assign_add(tf.reduce_sum(tf.abs(w))))

        # ProxSVRG update operations
        ops_svrg_update_v.append(v.assign(grad - grad_w0 + v0))

        # Iteration update
        ops_update_w.append(w.assign(prox_l1(w - lr*v, lbd*lr)))

        # update gradient mapping
        ops_update_grad_map.append(grad_map.assign((1/lr)*(w - prox_l1(w - lr*v0, lr * lbd))))
        
    # end for  

    ## group operations
    # Full gradient operations
    trainer_set_v0_to_zero  = tf.group(*ops_set_v0_to_zero)
    trainer_add_grad_to_v0  = tf.group(*ops_add_grad_to_v0)
    trainer_assign_w0       = tf.group(*ops_assign_w0)
    trainer_svrg_update_v   = tf.group(*ops_svrg_update_v)

    # calculate norm squares and l1 norm
    trainer_set_norm_v0_to_zero         = tf.group(*ops_set_norm_v0_to_zero)
    trainer_set_norm_v_to_zero          = tf.group(*ops_set_norm_v_to_zero)
    trainer_set_norm_grad_map_to_zero   = tf.group(*ops_set_norm_grad_map_to_zero)
    trainer_set_norm_l1_w_to_zero       = tf.group(*ops_set_norm_l1_w_to_zero)
    
    trainer_calc_norm_v0_sq              = tf.group(*ops_calc_norm_v0_sq)
    trainer_calc_norm_v_sq               = tf.group(*ops_calc_norm_v_sq)
    trainer_calc_norm_grad_map_sq        = tf.group(*ops_calc_norm_grad_map_sq)
    trainer_calc_norm_l1_w               = tf.group(*ops_calc_norm_l1_w)

    # ProxSVRG update operations
    trainer_update_w        = tf.group(*ops_update_w)
    trainer_update_grad_map = tf.group(*ops_update_grad_map)

    #=========================================#
    # ProxSVRG main algorithm

    # initialize history list
    hist_TrainLoss      = []
    hist_NumGrad        = []
    hist_NumEpoch       = []
    hist_GradNorm       = []
    hist_TrainAcc       = []
    hist_TestAcc        = []
    hist_MinGradNorm    = []
    w_sol               = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # calculate sample size
        num_examples = len(x_train)

        # setup batch size to compute full/batch gradient
        bs = 128
        num_batches_grad_full   = num_examples // bs

        scale_full  = 1/ (num_batches_grad_full + 0.0)
        
        # print initial message
        print("Training using Prox SVRG...")
        print('learning rate = {:.3e}'.format(LR), 'lambda = {:.3e}'.format(LBD), 'inner batch = ',inner_batch_size, '\n')
        
        # initialize stats variables
        min_grad_map_norm_square   = 1.0e6

        # variables used to update number of gradient evaluations
        num_grad    = 0
        num_epoch   = 0

        # store previous time when message had been printed
        last_print_num_grad = num_grad
        
        # ProxSVRG main loop
        while num_epoch < MAX_TOTAL_EPOCH:
            
            # save a copy of w to w0
            sess.run(ops_assign_w0)
            
            # Compute full gradient
            sess.run(trainer_set_v0_to_zero)
            for j in range(num_batches_grad_full): 
                batch_X = x_train[bs*j:bs*(j+1)]
                batch_Y = y_train[bs*j:bs*(j+1)]
                sess.run(trainer_add_grad_to_v0, feed_dict={x: batch_X, y: batch_Y, scale: scale_full})
                   
            if log_enable:
                # calculate loss, test accuracy
                sess.run(trainer_set_norm_l1_w_to_zero)
                sess.run(trainer_calc_norm_l1_w)
                train_loss = sess.run(loss_operation, feed_dict={x: x_train, y: y_train}) + LBD * sess.run(norm_l1_w)
                train_accuracy = sess.run(accuracy_operation, feed_dict = {x: x_train, y: y_train})
                test_accuracy = sess.run(accuracy_operation, feed_dict = {x: x_test, y: y_test})

                # compute gradient mapping
                sess.run(trainer_update_grad_map, feed_dict = {lr: LR_COMP, lbd: LBD})
                sess.run(trainer_set_norm_grad_map_to_zero)
                sess.run(trainer_calc_norm_grad_map_sq)
                grad_map_norm_square = sess.run(norm_grad_map_sq)

                # update mins
                if grad_map_norm_square < min_grad_map_norm_square:
                    min_grad_map_norm_square = grad_map_norm_square

                # print output
                if verbose:
                    print("Epoch:", num_epoch,"\nTraining Loss: ", train_loss)
                    print("Train Accuracy = {:.3f}".format(train_accuracy),"\nTest Accuracy = {:.3f}".format(test_accuracy))
                    print("||Gradient Mapping||^2 = {:.5e} ".format(grad_map_norm_square))
                    print("min ||Gradient Mapping||^2 = {:.5e} ".format(min_grad_map_norm_square), "\n")

                # update history
                hist_TrainLoss.append(train_loss)
                hist_GradNorm.append(np.asscalar(grad_map_norm_square))
                hist_MinGradNorm.append(min_grad_map_norm_square)
                hist_NumEpoch.append(num_epoch)
                hist_NumGrad.append(num_grad)
                hist_TrainAcc.append(train_accuracy)
                hist_TestAcc.append(test_accuracy)

                # update print time
                last_print_num_grad = num_grad

            # update number of gradient evaluations
            num_grad += num_examples
            num_epoch += 1
            
            # Inner loop and SVRG update  
            for inner in range(MAX_INNER_ITERATION):

                if inner_batch_size == 1:
                    # sample random index
                    i = np.random.randint(0,num_examples)

                    # extract sample
                    x_sample = np.expand_dims(x_train[i], axis=0)
                    y_sample = np.expand_dims(y_train[i], axis=0)

                    # SVRG Update
                    sess.run(trainer_svrg_update_v, feed_dict={x: x_sample, y: y_sample}) 
                    sess.run(trainer_update_w, feed_dict={lr: LR, lbd: LBD})
                    
                    
                else:
                    # sample mini batch
                    index = shuffle(np.array([id for id in range(num_examples)]), n_samples = inner_batch_size)

                    # extract samples
                    batch_x = x_train[index]
                    batch_y = y_train[index]

                    # SVRG Update
                    sess.run(trainer_svrg_update_v, feed_dict={x: batch_x, y: batch_y}) 
                    sess.run(trainer_update_w, feed_dict={lr: LR, lbd: LBD})                    

                # update number of gradient evaluations
                num_grad += 2*inner_batch_size
                num_epoch = num_grad / num_examples
            
        #end outer loop

        # calculate loss, test accuracy
        sess.run(trainer_set_norm_l1_w_to_zero)
        sess.run(trainer_calc_norm_l1_w)
        train_loss = sess.run(loss_operation, feed_dict={x: x_train, y: y_train}) + LBD * sess.run(norm_l1_w)
        train_accuracy = sess.run(accuracy_operation, feed_dict = {x: x_train, y: y_train})
        test_accuracy = sess.run(accuracy_operation, feed_dict = {x: x_test, y: y_test})

        # Compute full gradient
        sess.run(trainer_set_v0_to_zero)
        for j in range(num_batches_grad_full): 
            batch_X = x_train[bs*j:bs*(j+1)]
            batch_Y = y_train[bs*j:bs*(j+1)]
            sess.run(trainer_add_grad_to_v0, feed_dict={x: batch_X, y: batch_Y, scale: scale_full})
        
        # compute gradient mapping
        sess.run(trainer_update_grad_map, feed_dict = {lr: LR_COMP, lbd: LBD})
        sess.run(trainer_set_norm_grad_map_to_zero)
        sess.run(trainer_calc_norm_grad_map_sq)
        grad_map_norm_square = sess.run(norm_grad_map_sq)

        # update mins
        if grad_map_norm_square < min_grad_map_norm_square:
            min_grad_map_norm_square = grad_map_norm_square

        # print output
        if verbose:
            print("Epoch:", num_epoch,"\nTraining Loss: ", train_loss)
            print("Train Accuracy = {:.3f}".format(train_accuracy),"\nTest Accuracy = {:.3f}".format(test_accuracy))
            print("||Gradient Mapping||^2 = {:.5e} ".format(grad_map_norm_square))
            print("min ||Gradient Mapping||^2 = {:.5e} ".format(min_grad_map_norm_square), "\n")

        # update history
        hist_TrainLoss.append(train_loss)
        hist_GradNorm.append(np.asscalar(grad_map_norm_square))
        hist_MinGradNorm.append(min_grad_map_norm_square)
        hist_NumEpoch.append(num_epoch)
        hist_NumGrad.append(num_grad)
        hist_TrainAcc.append(train_accuracy)
        hist_TestAcc.append(test_accuracy)

        # save solution
        for w in w_list:
            w_sol.append(sess.run(w))

    return w_sol, hist_NumGrad, hist_NumEpoch, hist_TrainLoss, hist_GradNorm, hist_MinGradNorm, hist_TrainAcc, hist_TestAcc

#===============================================================================================================================
