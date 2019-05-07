import numpy as np

import sys
from src.utilities import *

from numpy.linalg import pinv
from numpy import identity as eye
from numpy import dot
from numpy.linalg import svd
import time

from sklearn.linear_model import Ridge as ridge
from scipy.optimize import fmin_cg

from training import *

def main(data_settings, training_settings):
    results, val_perf = training(data, data_settings, training_settings)
    return results, val_perf


if __name__ == "__main__":

    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    if len(sys.argv) > 1:
        seed = int(sys.argv[1])

        n_points = int(sys.argv[2])
        n_tasks = int(sys.argv[3])
        n_dims = int(sys.argv[4])
        method_IDX = int(sys.argv[5]) # 0 batch, 1 online
        dataset_IDX = int(sys.argv[6])
        if (method_IDX == 1):
            lambda_idx = int(sys.argv[7])
            c_value = float(sys.argv[8])
        else:
            lambda_idx = int(sys.argv[7])
            c_value = np.nan
    else:
        seed = 2
        n_points = 140
        n_tasks = 200
        n_dims = 50
        method_IDX = 1 # 0 batch, 1 online, 2 MTL, 3 ITL
        dataset_IDX = 0
        if (method_IDX == 1):
            lambda_idx = 999
            c_value = 35453
        else:
            lambda_idx = 999
            c_value = np.nan

    if method_IDX == 1:
        # c_value_range = [10 ** float(i) for i in np.linspace(-2, 8, 30)]




        # c_value_range = [10 ** float(i) for i in range(-4, 13)]

        c_value_range = [10 ** float(i) for i in range(1, 16)]
        # c_value_range = [10**8]




    elif (method_IDX == 3) or (method_IDX == 0) or (method_IDX == 2):
        c_value_range = [np.nan]




    if method_IDX == 2:
        lambda_range = [10 ** float(i) for i in np.linspace(-7, 3, 70)]
    else:
        lambda_range = [10 ** float(i) for i in np.linspace(-7, 2, 25)]
    # lambda_range = [10 ** float(i) for i in np.linspace(-5, -3, 25)]  # 50
    # lambda_range = [lambda_range[9]]
    # lambda_range = [10 ** float(i) for i in np.linspace(-8, 2, 4)]  # 50










    if lambda_idx != 999:
        lambda_range = [lambda_range[lambda_idx]]

    data_settings = {}
    data_settings['n_points'] = n_points
    data_settings['n_dims'] = n_dims
    data_settings['n_observed_tasks'] = n_tasks

    training_settings = {}
    # setting for step size on online LTL
    training_settings['conv_tol'] = 10 ** -4
    training_settings['param1_range'] = lambda_range


    if dataset_IDX == 0:
        data_settings['dataset'] = 'synthetic_regression'
    elif dataset_IDX == 1:
        data_settings['dataset'] = 'schools'



    for iiiiiiiiiiiii in range(1, 2):
        print('seed: %3d' % seed)
        data_settings['seed'] = seed
        # data generation
        if data_settings['dataset'] == 'synthetic_regression':
            data, data_settings = synthetic_data_gen(data_settings)
        elif data_settings['dataset'] == 'schools':
            data, data_settings = schools_data_gen(data_settings)

        if method_IDX == 0:
            training_settings['method'] = 'batch_LTL'
        elif method_IDX == 1:
            training_settings['method'] = 'online_LTL'
        elif method_IDX == 2:
            training_settings['method'] = 'MTL'
        elif method_IDX == 3:
            training_settings['method'] = 'Validation_ITL'

        for c_value_idx, c_value in enumerate(c_value_range):
            if dataset_IDX == 0:
                best_val_performance = 10 ** 8
            elif dataset_IDX == 1:
                best_val_performance = -10 ** 8

            for lambda_idx, param1 in enumerate(lambda_range):
                training_settings['c_value'] = c_value
                training_settings['param1'] = param1
                training_settings['param1idx'] = lambda_idx
                print('Working on lambda: %20.15f | c: %20.10f' % (param1, c_value))

                training_settings['filename'] = "seed_" + str(seed) + '-c_value_' + str(c_value)

                training_settings['foldername'] = 'results/' + data_settings['dataset'] + '-T_' + \
                                                  str(n_tasks) + '-n_' + str(n_points) + '/' \
                                                  + training_settings['method']

                results, val_perf = main(data_settings, training_settings)

                if dataset_IDX == 0:
                    if val_perf < best_val_performance:
                        validation_criterion = True
                    else:
                        validation_criterion = False
                elif dataset_IDX == 1:
                    if val_perf > best_val_performance:
                        validation_criterion = True
                    else:
                        validation_criterion = False

                if validation_criterion == True:
                    best_val_performance = val_perf
                    best_test_performance = results['test_perf']
                    best_param1 = param1
                    best_results = results

            # print(best_results)
            print('best param: %10e | best val error: %8.5f | best test error: %8.5f' %
                  (best_param1, best_val_performance, best_test_performance))
            save_results(best_results, data_settings, training_settings)

    print("done")
