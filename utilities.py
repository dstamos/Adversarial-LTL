import numpy as np
# import optimisation
import matplotlib.pylab as plt
import numpy.random as random
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from numpy.linalg import pinv
from numpy import identity as eye
from numpy.linalg import svd
from numpy.linalg import lstsq
from numpy.linalg import solve

import scipy as sp
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score


import os
import sys
import pickle
import time

def synthetic_data_gen(data_settings):
    # fixed generation
    np.random.seed(999)

    n_observed_tasks = data_settings['n_observed_tasks']
    n_test_tasks = 300
    n_tasks = n_test_tasks + n_observed_tasks

    data_settings['train_perc'] = 0.5
    data_settings['val_perc'] = 0.5
    data_settings['noise'] = 0.25
    # data_settings['noise'] = 0

    n_dims = data_settings['n_dims']
    n_points = data_settings['n_points']
    train_perc= data_settings['train_perc']
    val_perc = data_settings['val_perc']
    noise = data_settings['noise']


    # sparsity = round(0.5 * n_dims)
    # sparsity = round(0.8 * n_dims)
    sparsity = n_dims

    fixed_sparsity =  random.choice(np.arange(0, n_dims), sparsity, replace=False)
    # diagonal = np.zeros(n_dims)
    # diagonal[fixed_sparsity] = 1
    # D = np.zeros((n_dims, n_dims))
    # D[np.diag_indices_from(D)] = diagonal

    data = {}
    W_true = np.zeros((n_dims, n_tasks))
    X_train, Y_train = [None]*n_tasks, [None]*n_tasks
    X_val, Y_val = [None] * n_tasks, [None] * n_tasks
    X_test, Y_test = [None] * n_tasks, [None] * n_tasks
    for task_idx in range(0, n_tasks):
        # generating and normalizing the data
        features = random.randn(n_points, n_dims)
        features = features / norm(features, axis=1, keepdims=True)

        # generating and normalizing the weight vectors
        weight_vector = np.zeros((n_dims, 1))
        # weight_vector = 0.000001 * random.randn(n_dims, 1)
        weight_vector[fixed_sparsity] = random.randn(sparsity, 1)
        weight_vector = (weight_vector / norm(weight_vector)).ravel() * np.random.randint(1, 10)

        labels = features @ weight_vector + noise * random.randn(n_points)

        X_train_all, X_test[task_idx], Y_train_all, Y_test[task_idx] = train_test_split(features, labels, test_size=100)#, random_state = 42)
        test_size = int(np.floor(val_perc * len(Y_train_all)))
        X_train[task_idx], X_val[task_idx], Y_train[task_idx], Y_val[task_idx] = train_test_split(X_train_all, Y_train_all, test_size=test_size)

        W_true[:, task_idx] =  weight_vector







    # seeded split
    np.random.seed(data_settings['seed'])

    n_train_tasks = round(n_observed_tasks * 0.5)
    shuffled_tasks = np.random.permutation(n_observed_tasks+n_test_tasks)
    data_settings['task_range_tr'] = list(shuffled_tasks[:n_train_tasks])

    for task_idx in data_settings['task_range_tr']:
        X_temp = np.concatenate((X_val[task_idx], X_train[task_idx]))
        Y_temp = np.concatenate((Y_val[task_idx], Y_train[task_idx]))
        X_train[task_idx] = X_temp
        Y_train[task_idx] = Y_temp
        X_val[task_idx] = []
        Y_val[task_idx] = []

    data_settings['task_range_val'] = list(shuffled_tasks[n_train_tasks:n_observed_tasks])
    data_settings['task_range_test'] = list(shuffled_tasks[n_observed_tasks:n_observed_tasks + n_test_tasks])
    # n_tasks = n_train_tasks + n_val_tasks + n_test_tasks
    data_settings['task_range'] = list(np.arange(0, n_tasks))
    data_settings['n_tasks'] = n_tasks

    data['X_train'] = X_train
    data['Y_train'] = Y_train
    data['X_val'] = X_val
    data['Y_val'] = Y_val
    data['X_test'] = X_test
    data['Y_test'] = Y_test
    data['W_true'] = W_true

    return data, data_settings


def schools_data_gen(data_settings):
    np.random.seed(data_settings['seed'])

    n_observed_tasks = data_settings['n_observed_tasks']

    n_train_tasks = int(round(n_observed_tasks * 0.5))
    n_val_tasks = n_observed_tasks - n_train_tasks
    n_test_tasks = 139 - n_observed_tasks
    n_tasks = n_train_tasks+n_val_tasks+n_test_tasks

    task_shuffled = np.random.permutation(n_tasks)

    data_settings['task_range_tr'] = task_shuffled[0:n_train_tasks]
    data_settings['task_range_val'] = task_shuffled[n_train_tasks:n_train_tasks+n_val_tasks]
    data_settings['task_range_test'] = task_shuffled[n_train_tasks+n_val_tasks:]

    data_settings['task_range'] = task_shuffled
    data_settings['n_tasks'] = n_tasks

    temp = sio.loadmat('schoolData.mat')

    all_data = temp['X'][0]
    all_labels = temp['Y'][0]

    n_tasks = len(all_data)
    task_range_tr = data_settings['task_range_tr']
    task_range_val = data_settings['task_range_val']
    task_range_test = data_settings['task_range_test']


    X_train, Y_train = [None] * n_tasks, [None] * n_tasks
    X_val, Y_val = [None] * n_tasks, [None] * n_tasks
    X_test, Y_test = [None] * n_tasks, [None] * n_tasks

    # training tasks:
    max_score = 0
    for task_idx in task_range_tr:
        X_train[task_idx] = all_data[task_idx].T
        Y_train[task_idx] = all_labels[task_idx]

        X_train[task_idx] = X_train[task_idx] / norm(X_train[task_idx], axis=1, keepdims=True)

        Y_train[task_idx] = Y_train[task_idx].ravel()

        if max(Y_train[task_idx]) > max_score:
            max_score = max(Y_train[task_idx])

        #
        # miny = min(min of whatever)
        # maxy = max(max of whather)
        #
        # y in [ 0 1]
        # (y - miny) / (maxy - miny)

        X_val[task_idx] = []
        Y_val[task_idx] = []

        X_test[task_idx] = []
        Y_test[task_idx] = []
        # print('training tasks | n_points: %3d' % len(Y_train[task_idx]))

    for task_idx in task_range_val:
        X_train[task_idx], X_val[task_idx], Y_train[task_idx], Y_val[task_idx] = train_test_split(all_data[task_idx].T,
                                                                                                  all_labels[task_idx],
                                                                                                  test_size=0.75)

        Y_train[task_idx] = Y_train[task_idx].ravel()
        Y_val[task_idx] = Y_val[task_idx].ravel()

        if max(Y_train[task_idx]) > max_score:
            max_score = max(Y_train[task_idx])

        X_train[task_idx] = X_train[task_idx] / norm(X_train[task_idx], axis=1, keepdims=True)
        X_val[task_idx] = X_val[task_idx] / norm(X_val[task_idx], axis=1, keepdims=True)

        X_test[task_idx] = []
        Y_test[task_idx] = []
        # print('validation tasks | n_points training: %3d | n_points validation: %3d' % (len(Y_train[task_idx]), len(Y_val[task_idx])))


    for task_idx in task_range_tr:
        Y_train[task_idx] = Y_train[task_idx] / 1

    for task_idx in task_range_val:
        Y_train[task_idx] = Y_train[task_idx] / 1
        Y_val[task_idx] = Y_val[task_idx] / 1

    # max_score = 0
    for task_idx in task_range_test:
        X_train_all, X_test[task_idx], Y_train_all, Y_test[task_idx] = train_test_split(all_data[task_idx].T,
                                                                                        all_labels[task_idx],
                                                                                        test_size=0.5)
        X_train[task_idx], X_val[task_idx], Y_train[task_idx], Y_val[task_idx] = train_test_split(X_train_all,
                                                                                                  Y_train_all,
                                                                                                  test_size=0.75)
        Y_train[task_idx] = Y_train[task_idx].ravel()
        Y_val[task_idx] = Y_val[task_idx].ravel()
        Y_test[task_idx] = Y_test[task_idx].ravel()

        # if max(Y_train[task_idx]) > max_score:
        #     max_score = max(Y_train[task_idx])

        X_train[task_idx] = X_train[task_idx] / norm(X_train[task_idx], axis=1, keepdims=True)
        X_val[task_idx] = X_val[task_idx] / norm(X_val[task_idx], axis=1, keepdims=True)
        X_test[task_idx] = X_test[task_idx] / norm(X_test[task_idx], axis=1, keepdims=True)

        # print('test tasks | n_points training: %3d | n_points validation: %3d | n_points test: %3d' % (
        # len(Y_train[task_idx]), len(Y_val[task_idx]), len(Y_test[task_idx])))

    for task_idx in task_range_test:
        Y_train[task_idx] = Y_train[task_idx] / 1
        Y_val[task_idx] = Y_val[task_idx] / 1
        Y_test[task_idx] = Y_test[task_idx] / 1

    data_settings['n_dims'] = X_train[0].shape[1]

    data = {}
    data['X_train'] = X_train
    data['Y_train'] = Y_train
    data['X_val'] = X_val
    data['Y_val'] = Y_val
    data['X_test'] = X_test
    data['Y_test'] = Y_test
    return data, data_settings


def mean_squared_error(data_settings, X, true, W, task_indeces):
    if data_settings['dataset'] == 'synthetic_regression':
        n_tasks = len(task_indeces)
        mse = 0
        for _, task_idx in enumerate(task_indeces):
            n_points = len(true[task_idx])
            pred = X[task_idx] @ W[:, task_idx]

            mse = mse + norm(true[task_idx].ravel() - pred) ** 2 / n_points

        performance = mse / n_tasks
        mse = mse / n_tasks
    elif data_settings['dataset'] == 'schools':
        n_tasks = len(task_indeces)
        explained_variance = 0
        for _, task_idx in enumerate(task_indeces):
            n_points = len(true[task_idx])
            pred = X[task_idx] @ W[:, task_idx]

            # mse = norm(true[task_idx].ravel() - pred)**2 / n_points
            # explained_variance = explained_variance +  (1 - mse/np.var(true[task_idx]))
            explained_variance = explained_variance + explained_variance_score(true[task_idx].ravel(), pred)

        performance = 100 * explained_variance / n_tasks
    return performance


def batch_grad_func(D, task_indeces, data, switch):
    X = [data['X_train'][i] for i in task_indeces]
    Y = [data['Y_train'][i] for i in task_indeces]
    n_dims = X[0].shape[1]

    M = lambda D, n, t: X[t] @ D @ X[t].T + n * eye(n)

    grad = np.zeros((n_dims, n_dims))
    for idx, _ in enumerate(task_indeces):
        n_points = len(Y[idx])

        # tt = time.time()
        # invM = pinv(M(D, n_points, idx))
        # invM = lstsq(X[idx] @ D @ X[idx].T + n_points * eye(n_points), eye(n_points), rcond=None)[0]



        # MD = M(D, n_points, idx)
        # print('potato computed: %6.3f' % (time.time() - tt))

        Y[idx] = np.reshape(Y[idx], [1, len(Y[idx])])
        invM = sp.linalg.inv(M(D, n_points, idx))
        curr_grad = X[idx].T @ invM @ ((Y[idx].T @ Y[idx]) @ invM + invM @ (Y[idx].T @ Y[idx])) @ invM @ X[idx]

        # YY = np.multiply.outer(Y[idx].ravel(), Y[idx].ravel())
        # curr_grad = X[idx].T @ invM @ (YY @ invM + invM @ YY) @ invM @ X[idx]

        curr_grad = -n_points * curr_grad

        # half_grad = X[idx].T @ np.linalg.solve(MD.T, np.linalg.solve(MD,YY).T ).T

        # curr_grad2 = -n_points  *(half_grad  + half_grad.T)

        if switch == 1:
            # Lipschitz = 6
            # Lipschitz = (6 / (np.sqrt(n_points) * n_points ** 2)) * norm(X[idx], ord=2) ** 3
            # Lipschitz = (6 / n_points) * norm(X[idx], ord=2) ** 3
            # Lipschitz = 6 / norm(X[idx], ord=2)**3
            # Lipschitz = 6 / n_points
            # Lipschitz = 36
            # step_size  = 1 / Lipschitz
            # grad = grad + step_size * curr_grad

            grad = grad + curr_grad
        else:
            grad = grad + curr_grad
    return grad


def solve_wrt_w(D, X, Y, n_tasks, data, W_pred, task_range):
    for _, task_idx in enumerate(task_range):
        n_points = len(Y[task_idx])
        # replace pinv with np.linalg.solve or wahtever
        try:
            curr_w_pred = (D @ X[task_idx].T @
                           np.linalg.solve(X[task_idx] @ D @ X[task_idx].T + n_points * eye(n_points), Y[task_idx] ) ).ravel()
        except:
            ciao = 3# curr_w_pred = (D @ X[task_idx].T @ pinv(X[task_idx] @ D @ X[task_idx].T + n_points * eye(n_points)) @ Y[task_idx]).ravel()
        W_pred[:, task_idx] = curr_w_pred
    return W_pred


def solve_wrt_D(D, training_settings, data, X_train, Y_train, n_points, task_range, param1):
    batch_objective = lambda D: sum([n_points[i] * norm(sp.linalg.inv(X_train[i] @ D @ X_train[i].T +
                                                                      n_points[i] * eye(n_points[i])) @
                                                        Y_train[i]) ** 2 for i in task_range])
    # batch_objective = lambda D: sum([n_points[i] * norm(lstsq(X_train[i] @ D @ X_train[i].T + n_points[i] * eye(n_points[i]), Y_train[i], rcond=None)[0]) ** 2 for i in task_range])

    # Lipschitz = 6
    # step_size = 1 / Lipschitz

    D = np.eye(D.shape[0])

    batch_grad = lambda D: batch_grad_func(D, task_range, data, 1)

    curr_obj = batch_objective(D)

    objectives = []
    n_iter = 1999999
    curr_tol1 = 10**10
    curr_tol2 = 10 ** 10
    conv_tol_obj = 10**-5 # training_settings['conv_tol']
    conv_tol_grad = 10**-5
    c_iter = 0


    t = time.time()
    while (c_iter < n_iter) and ((curr_tol1 > conv_tol_obj)):
    # while (c_iter < n_iter) and (curr_tol1 > conv_tol_obj):
        prev_D = D
        prev_obj = curr_obj

        step_size = 10**18
        # step_size = 1/6
        grad = batch_grad(prev_D)

        temp_D = psd_trace_projection(prev_D - step_size * grad, 1/param1)
        temp_obj = batch_objective(temp_D)

        while temp_obj > (prev_obj + np.trace(grad.T @ (temp_D - prev_D)) +
                          1 / (2*step_size) * norm(prev_D - temp_D, ord='fro')**2):
        # while temp_obj > (prev_obj - 0.5 * step_size * norm(grad, ord='fro') ** 2):
            step_size = 0.5 * step_size

            temp_D = psd_trace_projection(prev_D - step_size * grad, 1/param1)
            temp_obj = batch_objective(temp_D)


        D = psd_trace_projection(prev_D - step_size * grad, 1/param1)

        curr_obj = batch_objective(D)
        objectives.append(curr_obj)

        curr_tol1 = abs(curr_obj - prev_obj) / prev_obj
        curr_tol2 = norm(grad, 'fro')
        c_iter = c_iter + 1

        if curr_obj > 1.001 * prev_obj:
            print('fucked')
            # break

        if curr_tol1 < 10**-14:
            break

        if (time.time() - t > 0):
            t = time.time()
            print("iter: %5d | obj: %12.8f | objtol: %10e | gradtol: %10e | step: %5.3e" %
                  (c_iter, curr_obj, curr_tol1, curr_tol2, step_size))

    # plt.plot(objectives)
    # plt.pause(0.01)

    print("iter: %5d | obj: %12.8f | objtol: %10e | gradtol: %10e | step: %5.3e" %
          (c_iter, curr_obj, curr_tol1, curr_tol2, step_size))

    return D


def fista_matrix(X_train, Y_train, training_settings, data_settings, n_points, task_range, param1):

    X = [X_train[i] for i in task_range]
    Y = [Y_train[i] for i in task_range]
    n_points = [n_points[i] for i in task_range]

    n_dims = data_settings['n_dims']
    n_tasks = len(task_range)

    # alpha = np.random.randn(n_dims, n_tasks)
    # W = np.random.randn(n_dims, n_tasks)

    alpha = np.zeros((n_dims, n_tasks))
    W = np.zeros((n_dims, n_tasks))

    from scipy.linalg import eigh
    Lipschitz = max([1/n_points[i] * eigh(X[i].T @ X[i], eigvals=(n_dims-1, n_dims-1))[0] for i in range(len(task_range))])[0]


    penalty = lambda x: param1 * norm(x, ord='nuc')
    prox = lambda x: np.sign(x) * np.maximum(abs(x) - param1 / Lipschitz, 0)
    loss = lambda x: sum([((0.5 * n_points[i]) * norm(Y[i] - X[i] @ x[:, i])**2) for i in range(len(task_range))])
    grad = lambda x: np.array([- (1/n_points[i]) * X[i].T @ (Y[i] - X[i] @ x[:, i]) for i in range(len(task_range))]).T

    curr_iter = 0
    prev_cost = 10**10
    conv_tol_obj = 10**-6 # training_settings['conv_tol']
    conv_tol_grad = 10 ** -3
    curr_cost = loss(W) + penalty(W)
    n_iter = 999999
    # n_iter = 200
    theta = 1
    conv_check_obj = np.Inf
    conv_check_grad = np.Inf
    objectives = []

    fista = 1

    t = time.time()
    while (curr_iter < n_iter) and ((conv_check_obj > conv_tol_obj) or (conv_check_grad > conv_tol_grad)):
        curr_iter = curr_iter + 1
        prev_cost = curr_cost

        prev_W = W

        ##############################
        if fista == 1:

            line_search = 0

            if line_search == 1:
                step_size = 100
            else:
                step_size = 1/Lipschitz
            gradient = grad(alpha)

            temp_search_point = alpha - step_size * gradient
            U, s, Vt = np.linalg.svd(temp_search_point)
            s = prox(s)
            S = np.zeros((U.shape[0], Vt.shape[1]))
            S[:len(s), :len(s)] = np.diag(s)
            W = U @ S @ Vt

            theta = (np.sqrt(theta ** 4 + 4 * theta ** 2) - theta ** 2) / 2
            rho = 1 - theta + np.sqrt(1 - theta)

            alpha = rho * W - (rho - 1) * prev_W
            if line_search == 1:
                temp_obj = loss(alpha) + penalty(alpha)

                while temp_obj > (prev_cost + np.trace(gradient.T @ (alpha - prev_W)) + 0.5 / step_size * norm(alpha - prev_W, ord='fro')**2):
                    step_size = 0.5 * step_size
                    temp_search_point = prev_W - step_size * gradient
                    U, s, Vt = np.linalg.svd(temp_search_point)
                    s = prox(s)
                    S = np.zeros((U.shape[0], Vt.shape[1]))
                    S[:len(s), :len(s)] = np.diag(s)
                    W = U @ S @ Vt

                    alpha = rho * W - (rho - 1) * prev_W

                    temp_obj = loss(alpha) + penalty(alpha)
                    if step_size < 10**-50:
                        break

            curr_cost = loss(alpha) + penalty(alpha)
            objectives.append(curr_cost)

        elif fista == 0:
            step_size = (1 / Lipschitz)
            gradient = grad(prev_W)
            search_point = prev_W - step_size * gradient

            U, s, Vt = np.linalg.svd(search_point)
            s = prox(s)
            S = np.zeros((U.shape[0], Vt.shape[1]))
            S[:len(s), :len(s)] = np.diag(s)
            W = U @ S @ Vt

            curr_cost = loss(W) + penalty(W)
            objectives.append(curr_cost)

            if curr_cost > prev_cost:
                print('objective did not descend')
        #############################



        conv_check_obj = abs(prev_cost - curr_cost) / prev_cost
        conv_check_grad = norm(gradient, 'fro')

        if curr_cost > prev_cost:
            k=1

        if (time.time() - t > 10):
            t = time.time()
            print("iter: %6d | obj: %12.8f | tol1: %10e | tol2: %10e | step: %6.4f" % (curr_iter, curr_cost, conv_check_obj, conv_check_grad, step_size))

    print("iter: %6d | obj: %12.8f | tol1: %10e | tol2: %10e | step: %6.4f" % (
    curr_iter, curr_cost, conv_check_obj, conv_check_grad, step_size))

    # plt.figure()
    # plt.plot(np.log10(objectives))
    # plt.title(str(param1))
    # plt.pause(0.001)

    return W


def solve_wrt_D_stochastic(D, training_settings, data, X_train, Y_train, n_points, task_range, param1, c_iter):
    batch_objective = lambda D: sum([n_points[i] * norm(pinv(X_train[i] @ D @ X_train[i].T + n_points[i] * eye(n_points[i])) @ Y_train[i]) ** 2 for i in task_range])
    batch_grad = lambda D: batch_grad_func(D, task_range, data, 0)

    c_value = training_settings['c_value']

    curr_obj = batch_objective(D)

    objectives = []

    n_points_for_step = np.array(n_points).astype('float')
    n_points_for_step[n_points_for_step == 0] = np.nan
    n_iter = 1
    # n_iter = np.ceil(1000 / np.sqrt(np.nanmean(n_points_for_step)))

    curr_tol = 10 ** 10
    conv_tol = 10 ** -8
    inner_iter = 0

    t = time.time()
    while (inner_iter < n_iter) and (curr_tol > conv_tol):
        inner_iter = inner_iter + 1
        prev_D = D
        prev_obj = curr_obj

        c_iter = c_iter + inner_iter
        step_size = c_value / np.sqrt(c_iter)
        # step_size = 1 / (param1 * np.sqrt(c_iter))
        D = prev_D - step_size * batch_grad(prev_D)

        D = psd_trace_projection(D, 1/param1)


        curr_obj = batch_objective(D)
        objectives.append(curr_obj)

        curr_tol = abs(curr_obj - prev_obj) / prev_obj

        if (time.time() - t > 111):
            t = time.time()
            # plt.plot(objectives, "b")
            # plt.pause(0.0001)
            print("iter: %5d | obj: %20.18f | tol: %20.18f" % (c_iter, curr_obj, curr_tol))

        # s, U = np.linalg.eigh(D)
        # s = [max(si,0) for si in s]
        # if sum(s) < 1/param1:
        #     pass
        # else:
        #     s = s/sum(s)*1/param1
        # D = U @ np.diag(s) @ U.T


    return D, c_iter


def psd_trace_projection(D, constraint):

    s, U = np.linalg.eigh(D)
    s = np.maximum(s, 0)

    if np.sum(s) < constraint:
        return U @ np.diag(s) @ U.T

    search_points = np.insert(s, 0, 0)
    low_idx = 0
    high_idx = len(search_points)-1

    obj = lambda vec, x: np.sum(np.maximum(vec - x, 0))

    while (low_idx <= high_idx):
        mid_idx = np.int(np.round((low_idx + high_idx) / 2))
        s_sum = obj(s, search_points[mid_idx])

        if (s_sum == constraint):
            s = np.sort(s)
            D_proj = U @ np.diag(s) @ U.T
            return D_proj
        elif (s_sum > constraint):
            low_idx = mid_idx + 1
        elif (s_sum < constraint):
            high_idx = mid_idx - 1

    if (s_sum > constraint):
        slope = (s_sum - obj(s, search_points[mid_idx+1])) / (search_points[mid_idx] - search_points[mid_idx+1])
        intercept = s_sum - slope * search_points[mid_idx]

        matching_point = (constraint - intercept) / slope
        s_sum = obj(s, matching_point)
    elif (s_sum < constraint):
        slope = (s_sum - obj(s, search_points[mid_idx-1])) / (search_points[mid_idx] - search_points[mid_idx-1])
        intercept = s_sum - slope * search_points[mid_idx]

        matching_point = (constraint - intercept) / slope
        s_sum = obj(s, matching_point)

    s = np.maximum(s - matching_point, 0)
    s = np.sort(s)
    D_proj = U @ np.diag(s) @ U.T

    return D_proj


def save_results(results, data_settings, training_settings):
    foldername = training_settings['foldername']
    filename = training_settings['filename']
    param1_range = training_settings['param1_range']

    if not os.path.exists(foldername):
        os.makedirs(foldername)
    f = open(foldername + '/' + filename + ".pckl", 'wb')
    pickle.dump(results, f)
    pickle.dump(data_settings, f)
    pickle.dump(training_settings, f)
    f.close()
