import numpy as np

import sys
from utilities import *

from numpy.linalg import pinv
from numpy import identity as eye
from numpy import dot
from numpy.linalg import svd
import time

from sklearn.linear_model import Ridge as ridge
from scipy.optimize import fmin_cg

def training(data, data_settings, training_settings):
    method = training_settings['method']

    # print("    seed: " + str(seed))
    # print("n_points: " + str(n_points))
    # print(" n_tasks: " + str(n_tasks))
    # print("  n_dims: " + str(n_dims))
    # print("  method: " + str(training_settings['method']))
    # print(" dataset: " + str(data_settings['dataset']))
    # print(" c_value: " + str(c_value))

    if method =='Validation_ITL':
        results, val_perf = validation_ITL(data, data_settings, training_settings)

    elif method == 'MTL':
        results, val_perf = mtl(data, data_settings, training_settings)

    elif method == 'batch_LTL':
        results, val_perf = batch_LTL(data, data_settings, training_settings)

    elif method == 'online_LTL':
        results, val_perf = online_LTL(data, data_settings, training_settings)

    return results, val_perf



def validation_ITL(data, data_settings, training_settings):
    param1 = training_settings['param1']
    n_tasks = data_settings['n_tasks']
    n_dims = data_settings['n_dims']
    task_range_test = data_settings['task_range_test']

    W_pred = np.zeros((n_dims, n_tasks))

    #####################################################
    # OPTIMISATION
    D = np.eye(n_dims,n_dims)/param1
    # D = psd_trace_projection(D, 1/param1)

    W_pred = solve_wrt_w(D, data['X_train'], data['Y_train'], n_tasks, data, W_pred, task_range_test)

    #####################################################
    # VALIDATION
    val_perf = mean_squared_error(data_settings, data['X_val'], data['Y_val'], W_pred, task_range_test)
    all_val_perf = val_perf

    #####################################################
    # TEST
    # W_pred = solve_wrt_w(D, data['X_train'], data['Y_train'], n_tasks, data, W_pred, task_range_test)
    test_perf = mean_squared_error(data_settings, data['X_test'], data['Y_test'], W_pred, task_range_test)
    all_test_perf = test_perf

    results = {}
    results['param1'] = param1
    results['val_perf'] = val_perf
    results['test_perf'] = test_perf
    results['all_val_perf'] = val_perf
    results['all_test_perf'] = test_perf


    print('ITL: param1: %8e | val MSE: %7.5f | test MSE: %7.5f' %
          (param1, all_val_perf, all_test_perf))

    return results, val_perf


def mtl(data, data_settings, training_settings):
    param1 = training_settings['param1']
    n_tasks = data_settings['n_tasks']
    n_dims = data_settings['n_dims']
    task_range_tr = data_settings['task_range_tr']
    task_range_val = data_settings['task_range_val']
    task_range_test = data_settings['task_range_test']
    all_tasks = data_settings['task_range']


    T = len(task_range_test)
    all_train_perf, all_val_perf, all_test_perf = [None] * T, [None] * T, [None] * T

    time_lapsed = [None] * T
    # D = random.randn(n_dims, n_dims)

    all_D = [None] * T

    D = np.eye(n_dims)

    curr_task_range_tr = []  ##################
    # curr_task_range_tr = task_range_tr ###
    # for pure_task_idx, new_tr_task in enumerate(task_range_tr):  ##################
    for pure_task_idx, new_tr_task in enumerate([0]):  ###
        t = time.time()
        W_pred = np.zeros((n_dims, n_tasks))


        #####################################################
        # OPTIMISATION
        X_train, Y_train = [None] * n_tasks, [None] * n_tasks
        n_points = [0] * n_tasks
        for _, task_idx in enumerate(task_range_test):
            X_train[task_idx] = data['X_train'][task_idx]
            Y_train[task_idx] = data['Y_train'][task_idx]
            n_points[task_idx] = len(Y_train[task_idx])

        D = solve_wrt_D(D, training_settings, data, X_train, Y_train, n_points, task_range_test, param1)

        time_lapsed[pure_task_idx] = time.time() - t
        # Check performance on ALL training tasks for this D
        # W_pred = solve_wrt_w(D, data['X_train'], data['Y_train'], n_tasks, data, W_pred, task_range_tr)
        # train_perf = mean_squared_error(data_settings, data['X_train'], data['Y_train'], W_pred, task_range_tr)
        # train_perf = np.nan
        # all_train_perf[pure_task_idx] = train_perf

        #####################################################
        # VALIDATION
        W_pred = solve_wrt_w(D, data['X_train'], data['Y_train'], n_tasks, data, W_pred, task_range_test)

        val_perf = mean_squared_error(data_settings, data['X_val'], data['Y_val'], W_pred, task_range_test)
        all_val_perf[pure_task_idx] = val_perf

        #####################################################
        # TEST
        X_train, Y_train = [None] * n_tasks, [None] * n_tasks
        for _, task_idx in enumerate(task_range_test):
            X_train[task_idx] = np.concatenate((data['X_val'][task_idx], data['X_train'][task_idx]))
            Y_train[task_idx] = np.concatenate((data['Y_val'][task_idx], data['Y_train'][task_idx]))

        W_pred = solve_wrt_w(D, X_train, Y_train, n_tasks, data, W_pred, task_range_test)
        test_perf = mean_squared_error(data_settings, data['X_test'], data['Y_test'], W_pred, task_range_test)
        all_test_perf[pure_task_idx] = test_perf

        # print("Batch LTL | best validation stats:")
        print('T: %3d (%3d) | lambda: %6e | val MSE: %7.5f | test MSE: %7.5f | norm D: %4.2f' %
              (new_tr_task, len(curr_task_range_tr), param1, all_val_perf[pure_task_idx], all_test_perf[pure_task_idx], norm(D)))
        # print("")

    results = {}
    # results['D'] = D

    results['param1'] = param1
    results['val_perf'] = val_perf
    results['test_perf'] = test_perf
    results['all_val_perf'] = all_val_perf
    results['all_test_perf'] = all_test_perf
    results['time_lapsed'] = time_lapsed
    results['rank_D'] = np.linalg.matrix_rank(D)

        # time.sleep(0.2)

    # plt.plot(all_test_perf)
    # plt.pause(0.01)

    # print("best validation stats:")
    print('Batch LTL: param1: %8e | val MSE: %7.5f | test MSE: %7.5f | norm D: %4.2f' %
          (param1, all_val_perf[pure_task_idx], all_test_perf[pure_task_idx], norm(D)))

    return results, test_perf


def batch_LTL(data, data_settings, training_settings):
    param1 = training_settings['param1']
    n_tasks = data_settings['n_tasks']
    n_dims = data_settings['n_dims']
    task_range_tr = data_settings['task_range_tr']
    task_range_val = data_settings['task_range_val']
    task_range_test = data_settings['task_range_test']

    T = len(task_range_tr)
    all_train_perf, all_val_perf, all_test_perf = [None] * T, [None] * T, [None] * T

    time_lapsed = [None] * T
    # D = random.randn(n_dims, n_dims)


    ################################################################################
    ################################################################################
    ################################################################################
    n_tasks_step = int(5)
    # task_range_tr_indeces = np.arange(0, T - 1, n_tasks_step)

    task_range_tr_indeces = np.arange(0, np.round(0.5*(T+1)).astype(int), n_tasks_step)
    task_range_tr_indeces = np.append(task_range_tr_indeces, np.arange(task_range_tr_indeces[-1] +
                                                                       2*n_tasks_step, T-1, 2*n_tasks_step))


    if T - 1 not in task_range_tr_indeces:
        task_range_tr_indeces = np.append(task_range_tr_indeces, T - 1)
    task_range_tr_untouched = task_range_tr
    task_range_tr = [task_range_tr[i] for i in task_range_tr_indeces]
    ################################################################################
    ################################################################################
    ################################################################################


    all_D = [None] * T

    D = np.eye(n_dims)

    curr_task_range_tr = []  ##################
    # curr_task_range_tr = task_range_tr ###
    # for pure_task_idx, new_tr_task in enumerate([0.0]):
    for pure_task_idx, new_tr_task in enumerate(task_range_tr):  ##################
    # for pure_task_idx, new_tr_task in enumerate([0]):  ###
        t = time.time()
        W_pred = np.zeros((n_dims, n_tasks))

        pure_task_idx = task_range_tr_indeces[pure_task_idx]
        curr_task_range_tr = task_range_tr_untouched[:pure_task_idx + 1]


        #####################################################
        # OPTIMISATION
        X_train, Y_train = [None] * n_tasks, [None] * n_tasks
        n_points = [0] * n_tasks
        for _, task_idx in enumerate(curr_task_range_tr):
            X_train[task_idx] = data['X_train'][task_idx]
            Y_train[task_idx] = data['Y_train'][task_idx]
            n_points[task_idx] = len(Y_train[task_idx])

        D = solve_wrt_D(D, training_settings, data, X_train, Y_train, n_points, curr_task_range_tr, param1)

        # all_D[pure_task_idx] = D
        # if pure_task_idx >= 1:
        #     D = np.average(all_D[:pure_task_idx], axis=0)

        # D_vec = np.reshape(D, [np.size(D)])
        # batch_objective_vec(D_vec, X_train, Y_train, n_points, curr_task_range_tr)
        # batch_grad_vec(D_vec, data, curr_task_range_tr)
        #
        # batch_obj_cg = lambda D_vec: batch_objective_vec(D_vec, X_train, Y_train, n_points, curr_task_range_tr)
        # batch_grad_cg = lambda D_vec: batch_grad_vec(D_vec, data, curr_task_range_tr)
        # D_vec = fmin_cg(batch_obj_cg, x0=D_vec, maxiter=10**10, fprime=batch_grad_cg, gtol=10**-12)
        # D = np.reshape(D_vec, [n_dims, n_dims])
        #
        # # projection on the 1/lambda trace norm ball
        # U, s, Vt = svd(D)  # eigen
        # # U, s = np.linalg.eig(D)
        # s = np.sign(s) * [max(np.abs(s[i]) - param1, 0) for i in range(len(s))]
        # D = U @ np.diag(s) @ Vt

        time_lapsed[pure_task_idx] = time.time() - t
        # Check performance on ALL training tasks for this D
        # W_pred = solve_wrt_w(D, data['X_train'], data['Y_train'], n_tasks, data, W_pred, task_range_tr)
        # train_perf = mean_squared_error(data_settings, data['X_train'], data['Y_train'], W_pred, task_range_tr)
        # train_perf = np.nan
        # all_train_perf[pure_task_idx] = train_perf

        #####################################################
        # VALIDATION
        W_pred = solve_wrt_w(D, data['X_train'], data['Y_train'], n_tasks, data, W_pred, task_range_val)

        val_perf = mean_squared_error(data_settings, data['X_val'], data['Y_val'], W_pred, task_range_val)
        all_val_perf[pure_task_idx] = val_perf

        #####################################################
        # TEST
        X_train, Y_train = [None] * n_tasks, [None] * n_tasks
        for _, task_idx in enumerate(task_range_test):
            X_train[task_idx] = np.concatenate((data['X_val'][task_idx], data['X_train'][task_idx]))
            Y_train[task_idx] = np.concatenate((data['Y_val'][task_idx], data['Y_train'][task_idx]))

        W_pred = solve_wrt_w(D, X_train, Y_train, n_tasks, data, W_pred, task_range_test)
        test_perf = mean_squared_error(data_settings, data['X_test'], data['Y_test'], W_pred, task_range_test)
        all_test_perf[pure_task_idx] = test_perf

        # print("Batch LTL | best validation stats:")
        print('T: %3d (%3d) | lambda: %6e | val MSE: %7.5f | test MSE: %7.5f | norm D: %4.2f' %
              (new_tr_task, len(curr_task_range_tr), param1, all_val_perf[pure_task_idx], all_test_perf[pure_task_idx], norm(D)))
        # print("")

    results = {}
    # results['D'] = D

    results['param1'] = param1
    results['val_perf'] = val_perf
    results['test_perf'] = test_perf
    results['all_val_perf'] = all_val_perf
    results['all_test_perf'] = all_test_perf
    results['time_lapsed'] = time_lapsed
    results['rank_D'] = np.linalg.matrix_rank(D)

        # time.sleep(0.2)

    # plt.plot(all_test_perf)
    # plt.pause(0.01)

    # print("best validation stats:")
    print('Batch LTL: param1: %8e | val MSE: %7.5f | test MSE: %7.5f | norm D: %4.2f' %
          (param1, all_val_perf[pure_task_idx], all_test_perf[pure_task_idx], norm(D)))

    return results, val_perf


def online_LTL(data, data_settings, training_settings):
    param1 = training_settings['param1']
    param1idx = training_settings['param1idx']
    n_tasks = data_settings['n_tasks']
    n_dims = data_settings['n_dims']
    task_range_tr = data_settings['task_range_tr']
    task_range_val = data_settings['task_range_val']
    task_range_test = data_settings['task_range_test']

    T = len(task_range_tr)
    all_train_perf, all_val_perf, all_test_perf = [[] for i in range(T)], [[] for i in range(T)], [[] for i in range(T)]
    best_param, best_train_perf, best_val_perf, best_test_perf = [None] * T, [None] * T, [None] * T, [None] * T
    time_lapsed = [None] * T

    c_iter = 0
    D = np.eye(n_dims)
    # D = np.zeros((n_dims, n_dims))

    all_D = [None] * T
    all_ranks = [None] * T

    # plt.figure()

    W_pred = np.zeros((n_dims, n_tasks))
    for pure_task_idx, curr_task_range_tr in enumerate(task_range_tr):
        t = time.time()

        # OPTIMISATION
        X_train, Y_train = [None] * n_tasks, [None] * n_tasks
        n_points = [0] * n_tasks
        for _, task_idx in enumerate([curr_task_range_tr]):
            X_train[task_idx] = data['X_train'][task_idx]
            Y_train[task_idx] = data['Y_train'][task_idx]
            n_points[task_idx] = len(Y_train[task_idx])

        D, c_iter = solve_wrt_D_stochastic(D, training_settings, data, X_train, Y_train, n_points,
                                           [curr_task_range_tr], param1, c_iter)


        all_D[pure_task_idx] = D
        if pure_task_idx >= 1:
            D_average = np.average(all_D[:pure_task_idx], axis=0)
        else:
            D_average = D

        time_lapsed[pure_task_idx] = time.time() - t

        # print('T: %3d (%3d) | lambda: %6e | val MSE: %7.5f | test MSE: %7.5f' %
        #       (pure_task_idx, curr_task_range_tr, param1, np.nan, np.nan))


        # Check performance on ALL training tasks for this D
        # W_pred = solve_wrt_w(D, data['X_train'], data['Y_train'], n_tasks, data, W_pred, task_range_tr)
        # train_perf = mean_squared_error(data_settings, data['X_train'], data['Y_train'], W_pred, task_range_tr)
        # all_train_perf[pure_task_idx].append(train_perf)

        # individual_tr_perf = [None] * len(task_range_tr)
        # for idx, task_idx in enumerate(task_range_tr):
        #     individual_tr_perf[idx] = mean_squared_error(data_settings, data['X_train'], data['Y_train'], W_pred, [task_idx])

        #####################################################
        # VALIDATION
        W_pred = solve_wrt_w(D_average, data['X_train'], data['Y_train'], n_tasks, data, W_pred, task_range_val)

        val_perf = mean_squared_error(data_settings, data['X_val'], data['Y_val'], W_pred, task_range_val)
        all_val_perf[pure_task_idx].append(val_perf)

        #####################################################
        # TEST
        X_train, Y_train = [None] * n_tasks, [None] * n_tasks
        for _, task_idx in enumerate(task_range_test):
            X_train[task_idx] = np.concatenate((data['X_val'][task_idx], data['X_train'][task_idx]))
            Y_train[task_idx] = np.concatenate((data['Y_val'][task_idx], data['Y_train'][task_idx]))

        W_pred = solve_wrt_w(D_average, X_train, Y_train, n_tasks, data, W_pred, task_range_test)

        test_perf = mean_squared_error(data_settings, data['X_test'], data['Y_test'], W_pred, task_range_test)
        all_test_perf[pure_task_idx].append(test_perf)

        # best_train_perf[pure_task_idx] = train_perf
        best_val_perf[pure_task_idx] = val_perf
        best_test_perf[pure_task_idx] = test_perf
        # all_individual_tr_perf[pure_task_idx] = individual_tr_perf

        # U, s, V, = np.linalg.svd(D_pure)
        # all_ranks[pure_task_idx] = len(s[s > 10**-6])
    #
    # plt.imshow(D)
    # plt.pause(0.5)
    # plt.close()


    print('T: %3d (%3d) | lambda: %6e | val MSE: %7.5f | test MSE: %7.5f' %
          (pure_task_idx, curr_task_range_tr, param1, val_perf, test_perf))

    results = {}
    # results['all_train_perf'] = best_train_perf
    # results['all_individual_tr_perf'] = all_individual_tr_perf
    results['param1'] = param1
    results['val_perf'] = val_perf
    results['test_perf'] = test_perf
    results['all_val_perf'] = best_val_perf
    results['all_test_perf'] = best_test_perf
    results['time_lapsed'] = time_lapsed
    results['rank_D'] = np.linalg.matrix_rank(D)

    # if param1idx == 9:
    #     k = 1

    return results, val_perf
    # plt.plot(best_test_perf)
    # plt.pause(0.01)


def mtl_old(data, data_settings, training_settings):
    param1 = training_settings['param1']
    n_tasks = data_settings['n_tasks']
    n_dims = data_settings['n_dims']
    task_range_test = data_settings['task_range_test']
    all_tasks = data_settings['task_range']

    task_range_test = np.sort(task_range_test)

    T = len(task_range_test)

    time_lapsed = [None] * T
    # D = random.randn(n_dims, n_dims)

    n_tasks = len(all_tasks)


    t = time.time()
    W_pred = np.zeros((n_dims, n_tasks))

    #####################################################
    # OPTIMISATION
    X_train, Y_train = [None] * n_tasks, [None] * n_tasks
    n_points = [0] * n_tasks
    for _, task_idx in enumerate(task_range_test):
        X_train[task_idx] = data['X_train'][task_idx]
        Y_train[task_idx] = data['Y_train'][task_idx]
        n_points[task_idx] = len(Y_train[task_idx])

    W = fista_matrix(data['X_train'], data['Y_train'], training_settings, data_settings, n_points, task_range_test, param1)
    W_pred[:, task_range_test] = W

    time_lapsed = time.time() - t
    # Check performance on ALL training tasks for this D
    train_perf = mean_squared_error(data_settings, data['X_train'], data['Y_train'], W_pred, task_range_test)
    all_train_perf = train_perf

    #####################################################
    # VALIDATION
    val_perf = mean_squared_error(data_settings, data['X_val'], data['Y_val'], W_pred, task_range_test)
    all_val_perf = val_perf

    #####################################################
    # TEST
    test_perf = mean_squared_error(data_settings, data['X_test'], data['Y_test'], W_pred, task_range_test)
    all_test_perf = test_perf

    results = {}
    # results['D'] = D
    results['param1'] = param1
    results['val_perf'] = val_perf
    results['test_perf'] = test_perf
    results['all_val_perf'] = val_perf
    results['all_test_perf'] = test_perf

    print('MTL: param1: %8e | val MSE: %7.5f | test MSE: %7.5f' %
          (param1, all_val_perf, all_test_perf))

    return results, val_perf