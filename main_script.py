import numpy as np
import matplotlib.pyplot as plt
from src.settings import DataSettings, TrainingSettings
from src.data_handler import DataHandler
from src.save import Logger
from src.training import LearningToLearnD, IndipendentTaskLearning
from src.multitask_learning import MultitaskLearning
from copy import deepcopy


if __name__ == "__main__":
    np.random.seed(999)

    data_info_dict = {'dataset': 'synthetic',
                      'n_tr_tasks': 30,
                      'n_val_tasks': 2,
                      'n_test_tasks': 20,
                      'n_all_points': 80,
                      'ts_points_pct': 0.5,
                      'n_dims': 20,
                      'noise_std': 0.2,
                      'seed': 999}

    # data_info_dict = {'dataset': 'synthetic_data_gen_biased_sgd_paper',
    #                   'n_tr_tasks': 200,
    #                   'n_val_tasks': 2,
    #                   'n_test_tasks': 20,
    #                   'n_all_points': 20,
    #                   'ts_points_pct': 0.5,
    #                   'n_dims': 30,
    #                   'noise_std': 0.1,
    #                   'seed': 999}

    # data_info_dict = {'dataset': 'schools',
    #                   'n_tr_tasks': 50,
    #                   'n_val_tasks': 50,
    #                   'n_test_tasks': 39,
    #                   'ts_points_pct': 0.25,
    #                   'seed': 999}

    # data_info_dict = {'dataset': 'movielens100k',
    #                   'n_tr_tasks': 400,
    #                   'n_val_tasks': 20,
    #                   'n_test_tasks': 343,
    #                   'ts_points_pct': 0.25,
    #                   'seed': 999}

    # training_info_dict = {'method': 'temp_method',
    #                       'inner_regul_param': [10 ** float(i) for i in np.linspace(-6, 3, 10)],
    #                       'meta_algo_regul_param': [10 ** float(i) for i in np.linspace(-4, 3, 10)],
    #                       'convergence_tol': 10 ** -4}
    # training_info_dict['inner_regul_param'] = [training_info_dict['inner_regul_param'][6]]
    # training_info_dict['meta_algo_regul_param'] = [training_info_dict['meta_algo_regul_param'][4]]

    training_info_dict = {'method': 'indipendent',
                          'inner_regul_param': [10 ** float(i) for i in np.linspace(-8, 4, 30)],
                          'meta_algo_regul_param': [np.nan],
                          'convergence_tol': 10 ** -4}
    # training_info_dict['meta_algo_regul_param'] = [training_info_dict['meta_algo_regul_param'][3]]
    # training_info_dict['inner_regul_param'] = [training_info_dict['inner_regul_param'][5]]

    # training_info_dict['meta_algo_regul_param'] = [training_info_dict['meta_algo_regul_param'][5]]
    # training_info_dict['inner_regul_param'] = [training_info_dict['inner_regul_param'][15]]

    # training_info_dict = {'method': 'multitask',
    #                       'inner_regul_param': [10 ** float(i) for i in np.linspace(-6, 3, 20)],
    #                       'meta_algo_regul_param': [np.nan],
    #                       'convergence_tol': 10 ** -4}

    data_info = DataSettings(data_info_dict)
    training_info = TrainingSettings(training_info_dict)
    data = DataHandler(data_info)

    # model = LearningToLearnD(data_info, logger, verbose=1)
    # model.fit(data)
    # exit()

    best_val_score = np.Inf
    all_val_scores = np.zeros((len(training_info.inner_regul_param), len(training_info.meta_algo_regul_param)))
    all_test_scores = np.zeros((len(training_info.inner_regul_param), len(training_info.meta_algo_regul_param)))
    for inner_regul_param_idx, inner_regul_param in enumerate(training_info.inner_regul_param):
        for meta_algo_regul_param_idx, meta_algo_regul_param in enumerate(training_info.meta_algo_regul_param):
            logger = Logger(data_info, training_info, inner_regul_param, meta_algo_regul_param)

            if training_info.method == 'temp_method':
                model = LearningToLearnD(data_info, logger, meta_algo_regul_param=meta_algo_regul_param, inner_regul_param=inner_regul_param, verbose=1)
            elif training_info.method == 'indipendent':
                model = IndipendentTaskLearning(data_info, logger, meta_algo_regul_param=meta_algo_regul_param, inner_regul_param=inner_regul_param, verbose=1)
            elif training_info.method == 'multitask':
                model = MultitaskLearning(data_info, logger, meta_algo_regul_param=meta_algo_regul_param, inner_regul_param=inner_regul_param, verbose=1)
            else:
                raise ValueError("Unknown model.")

            model.fit(data)
            logger.save(model.results)

            if model.results['val_score'] < best_val_score:
                best_model = deepcopy(model)
            all_val_scores[inner_regul_param_idx, meta_algo_regul_param_idx] = model.results['val_score']
            all_test_scores[inner_regul_param_idx, meta_algo_regul_param_idx] = np.average(model.results['test_scores'])

    k = 1
