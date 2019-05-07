import numpy as np
from src.settings import DataSettings, TrainingSettings
from src.data_handler import DataHandler
from src.save import Logger
from src.training import LearningToLearnD, mtl_mse_scorer
from sklearn.model_selection import GridSearchCV


if __name__ == "__main__":
    data_info_dict = {'dataset': 'synthetic',
                      'n_all_tasks': 300,
                      'n_tr_tasks': 100,
                      'n_val_tasks': 50,
                      'n_all_points': 100,
                      'val_points_pct': 25,
                      'ts_points_pct': 75,
                      'n_dims': 50,
                      'noise_std': 0.25,
                      'seed': 999}

    training_info_dict = {'method': 'temp_method',
                          'c_value_range': [10 ** float(i) for i in range(1, 16)],
                          'lambda_range': [10 ** float(i) for i in np.linspace(-7, 3, 70)],
                          'convergence_tol': 10 ** -4}

    data_info = DataSettings(data_info_dict)
    training_info = TrainingSettings(training_info_dict)
    data = DataHandler(data_info)
    logger = Logger(data_info, training_info)

    model = GridSearchCV(LearningToLearnD(verbose=1),
                         cv=[(data.tr_task_indexes, data.val_task_indexes)],
                         param_grid={"lambda_value": training_info.lambda_range,
                                     "c_value": training_info.c_value_range},
                         scoring=mtl_mse_scorer,
                         n_jobs=-2,
                         verbose=10)
    model.fit(data)
    k = 1

    # model = ModelSelection(data, training_info)
    # model.fit(data.features_tr, data.labels_tr)

    k = 1
