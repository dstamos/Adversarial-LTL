import numpy as np
import sys
from src.settings import DataSettings, TrainingSettings
from src.data_handler import DataHandler
from src.save import Logger
from src.training import LearningToLearnD, IndipendentTaskLearning, AverageRating


if __name__ == "__main__":
    np.random.seed(999)

    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        dataset_idx = int(sys.argv[2])
        method_idx = int(sys.argv[3])
        inner_param_idx = int(sys.argv[4])
        meta_param_idx = int(sys.argv[5])
    else:
        seed = 999
        dataset_idx = 3  # 0: synthetic, 1: schools, 2: movielens100k, 3: miniwikipedia
        method_idx = 0  # 0: ITL_SGD, 1: ITL_ERM, 2: LTL_SGD-SGD, 3: LTL_ERM-SGD, 4: LTL_Oracle-SGD
        inner_param_idx = 9
        meta_param_idx = 3

    np.random.seed(seed)
    inner_regul_param_range = [10 ** float(i) for i in np.linspace(-6, 3, 20)]
    meta_regul_param_range = [10 ** float(i) for i in np.linspace(-4, 3, 10)]

    if dataset_idx == 0:
        data_info_dict = {'dataset': 'synthetic',
                          'n_tr_tasks': 3000,
                          'n_val_tasks': 300,
                          'n_test_tasks': 300,
                          'n_all_points': 80,
                          'ts_points_pct': 0.5,
                          'n_dims': 20,
                          'noise_std': 0.2,
                          'seed': seed}
    elif dataset_idx == 1:
        data_info_dict = {'dataset': 'schools',
                          'n_tr_tasks': 50,
                          'n_val_tasks': 50,
                          'n_test_tasks': 39,
                          'ts_points_pct': 0.25,
                          'seed': seed}
    elif dataset_idx == 2:
        data_info_dict = {'dataset': 'movielens100k',
                          'n_tr_tasks': 500,
                          'n_val_tasks': 100,
                          'n_test_tasks': 343,
                          'ts_points_pct': 0.25,
                          'seed': seed}
    elif dataset_idx == 3:
        data_info_dict = {'dataset': 'miniwikipedia',
                          'n_tr_tasks': 500,  # 813 total
                          'n_val_tasks': 100,
                          'n_test_tasks': 213,
                          'ts_points_pct': 0.25,
                          'seed': seed}
    else:
        raise ValueError('Unknown dataset.')

    if method_idx == 0:
        training_info_dict = {'method': 'ITL_SGD',
                              'inner_regul_param': inner_regul_param_range[inner_param_idx],
                              'meta_algo_regul_param': np.nan}
    elif method_idx == 1:
        training_info_dict = {'method': 'ITL_ERM',
                              'inner_regul_param': inner_regul_param_range[inner_param_idx],
                              'meta_algo_regul_param': np.nan}
    elif method_idx == 2:
        training_info_dict = {'method': 'LTL_SGD-SGD',
                              'inner_regul_param': inner_regul_param_range[inner_param_idx],
                              'meta_algo_regul_param': meta_regul_param_range[meta_param_idx]}
    elif method_idx == 3:
        training_info_dict = {'method': 'LTL_ERM-SGD',
                              'inner_regul_param': inner_regul_param_range[inner_param_idx],
                              'meta_algo_regul_param': meta_regul_param_range[meta_param_idx]}
    elif method_idx == 4:
        training_info_dict = {'method': 'LTL_Oracle-SGD',
                              'inner_regul_param': inner_regul_param_range[inner_param_idx],
                              'meta_algo_regul_param': meta_regul_param_range[meta_param_idx]}
    else:
        raise ValueError('Unknown method.')

    data_info = DataSettings(data_info_dict)
    training_info = TrainingSettings(training_info_dict)
    data = DataHandler(data_info)

    logger = Logger(data_info, training_info, training_info.inner_regul_param, training_info.meta_algo_regul_param)

    if training_info.method == 'LTL_SGD-SGD' or training_info.method == 'LTL_ERM-SGD' or training_info.method == 'LTL_Oracle-SGD':
        model = LearningToLearnD(data_info, logger, training_info, verbose=1)
    elif training_info.method == 'ITL_SGD' or training_info.method == 'ITL_ERM':
        if data_info.dataset == 'movielens100k':
            model = AverageRating(data_info, logger, training_info, verbose=1)
        else:
            model = IndipendentTaskLearning(data_info, logger, training_info, verbose=1)
    else:
        raise ValueError("Unknown method.")

    model.fit(data)
    logger.save(model.results)
