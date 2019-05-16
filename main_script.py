import numpy as np
import matplotlib.pyplot as plt
from src.settings import DataSettings, TrainingSettings
from src.data_handler import DataHandler
from src.save import Logger
from src.training import LearningToLearnD
from copy import deepcopy


if __name__ == "__main__":
    np.random.seed(999)

    data_info_dict = {'dataset': 'synthetic',
                      'n_all_tasks': 300,
                      'n_tr_tasks': 150,
                      'n_val_tasks': 500,
                      'n_all_points': 40,
                      'ts_points_pct': 0.5,
                      'n_dims': 50,
                      'noise_std': 0.2,
                      'seed': 999}

    training_info_dict = {'method': 'temp_method',
                          'inner_regul_param': [10 ** float(i) for i in np.linspace(-1.5, 1, 5)],
                          'meta_algo_regul_param': [10 ** float(i) for i in np.linspace(-1.5, 2, 10)],
                          'convergence_tol': 10 ** -4}

    data_info = DataSettings(data_info_dict)
    training_info = TrainingSettings(training_info_dict)
    data = DataHandler(data_info)
    logger = Logger(data_info, training_info)

    model = LearningToLearnD(data_info, logger, verbose=1)
    model.fit(data)

    # best_val_score = np.Inf
    # all_val_scores = np.zeros((len(training_info.inner_regul_param), len(training_info.meta_algo_regul_param)))
    # for inner_regul_param_idx, inner_regul_param in enumerate(training_info.inner_regul_param):
    #     for meta_algo_regul_param_idx, meta_algo_regul_param in enumerate(training_info.meta_algo_regul_param):
    #         model = LearningToLearnD(data_info, logger, meta_algo_regul_param=meta_algo_regul_param, inner_regul_param=inner_regul_param, verbose=1)
    #
    #         model.fit(data)
    #
    #         if model.results['val_score'] < best_val_score:
    #             best_model = deepcopy(model)
    #             all_val_scores[inner_regul_param_idx, meta_algo_regul_param_idx] = model.results['val_score']
    # plt.imshow(all_val_scores)
    # plt.colorbar()
    # plt.title('val errors')
    # plt.pause(0.1)
    k = 1

    # model = ModelSelection(data, training_info)
    # model.fit(data.features_tr, data.labels_tr)

    k = 1
