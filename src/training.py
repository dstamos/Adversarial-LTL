import numpy as np
from sklearn.metrics import mean_squared_error


class LearningToLearnD:
    def __init__(self, lambda_value=1e-3, c_value=10, data_info=None, verbose=1):
        self.verbose = verbose
        self.data_info = data_info
        self.lambda_value = lambda_value
        self.c_value = c_value

        self.D = None
        self.w = [None]

    def fit(self, task_list):
        # TODO Add the LTL optimization here and put the D matrix in the bucket self
        self.D = np.random.randn(self.data_info.n_dims, self.data_info.n_dims)

    def predict(self, input_list):
        # TODO based on self.D optimize for w and recover predictions
        n_tasks = len(input_list)
        self.w = [None] * n_tasks
        for task_idx in range(n_tasks):
            self.w[task_idx] = np.random.randn(self.data_info.n_dims)

    def get_params(self, deep=True):
        return {"lambda_value": self.lambda_value, "c_value": self.c_value}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def mtl_mse_scorer(output_pred, output_true):
    n_tasks = len(output_true)

    metric = 0
    for task_idx in range(n_tasks):
        c_metric = mean_squared_error(output_true[task_idx], output_pred[task_idx])
        metric = metric + c_metric
    metric = metric / n_tasks

    return metric
