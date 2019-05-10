import numpy as np
import scipy as sp
from sklearn.metrics import mean_squared_error


class LearningToLearnD:
    def __init__(self, data_info, logger, meta_algo_regul_param=1e-1, inner_regul_param=10, verbose=1):
        self.verbose = verbose
        self.data_info = data_info
        self.logger = logger
        self.meta_algo_regul_param = meta_algo_regul_param
        self.inner_regul_param = inner_regul_param

        self.representation_d = None

    def fit(self, data):
        n_dims = data.data_info.n_dims

        test_scores = []

        curr_theta = np.zeros((n_dims, n_dims))
        curr_representation_d = np.eye(n_dims) / n_dims
        representation_d = curr_representation_d
        for task_idx, task in enumerate(data.tr_task_indexes):
            prev_theta = curr_theta

            approx_grad, weight_vector = self.inner_algo(curr_representation_d, data.features_tr[task], data.labels_tr[task])

            # Update Theta
            curr_theta = prev_theta + approx_grad

            # Compute M
            matrix_m = np.exp(-curr_theta/self.meta_algo_regul_param)

            # Rescale
            curr_representation_d = matrix_m / np.trace(matrix_m)

            # Average:
            representation_d = (representation_d * (task_idx + 1) + curr_representation_d * 1) / (task_idx + 2)
            self.representation_d = representation_d

            # TODO Test error (maybe validation?), logging, printouts
            predictions_ts = []
            for test_task_idx, test_task in enumerate(data.test_task_indexes):
                # print(test_task_idx)
                predictions_ts.append(self.predict(representation_d, data.features_ts[test_task], data.labels_ts[test_task]))
            test_scores.append(mtl_mse_scorer(predictions_ts, [data.labels_ts[i] for i in data.test_task_indexes]))

            printout = "T: %(task)3d | test score: %(ts_score)6.4f" % {'task': task, 'ts_score': float(np.mean(test_scores))}
            self.logger.log_event(printout)

        # TODO Optimization wrt w on val tasks, training points
        # Reuse the same function as Algo 1 or 2 but just return w
        for task_idx, task in enumerate(data.val_task_indexes):
            _, weight_vector = self.inner_algo(self.representation_d, data.features_ts[task], data.labels_ts[task])

        # TODO Performance check of (D, w) on val tasks, validation points
        # Call MSE

    def inner_algo(self, representation_d, features, labels, inner_algo_method='algo_1'):
        def absolute_loss(curr_features, curr_labels, weight_vector):
            n_points = curr_features.shape[0]
            loss = 0
            for point_idx in range(n_points):
                loss = loss + np.abs(curr_labels[point_idx] - curr_features[point_idx, :] @ weight_vector)
            return loss / n_points

        if inner_algo_method == 'algo_1':
            def penalty(weight_vector, n_points):
                return self.inner_regul_param / (2*n_points) * np.linalg.norm(weight_vector, ord=2) ** 2
        elif inner_algo_method == 'algo_2':
            def penalty(weight_vector):
                return self.inner_regul_param * np.linalg.norm(weight_vector, ord=2)**2
        else:
            raise ValueError("Unknown inner algorithm.")

        curr_u = np.zeros(self.data_info.n_dims)
        curr_w_d = np.zeros(self.data_info.n_dims)
        moving_average_approx_subgradient = curr_u
        moving_average_weights = curr_w_d
        for curr_point_idx in range(features.shape[0]):
            prev_u = curr_u

            pred = sp.linalg.sqrtm(representation_d) @ features[curr_point_idx, :] @ prev_u
            true = labels[curr_point_idx]
            if true > pred:
                u = true - 1
            else:
                u = -(true - 1)
            if inner_algo_method == 'algo_1':
                curr_u = prev_u - 1 / self.inner_regul_param * sp.linalg.sqrtm(representation_d) @ features[curr_point_idx, :] * u
            elif inner_algo_method == 'algo_2':
                curr_u = prev_u - 1 / self.inner_regul_param * (sp.linalg.sqrtm(representation_d) @ features[curr_point_idx, :] * u + self.inner_regul_param * prev_u)
            else:
                raise ValueError("Unknown inner algorithm.")
            curr_w_d = sp.linalg.sqrtm(representation_d) @ curr_u

            moving_average_weights = (moving_average_weights * (curr_point_idx + 1) + curr_w_d * 1) / (curr_point_idx + 2)
            moving_average_approx_subgradient = (moving_average_approx_subgradient * (curr_point_idx + 1) + curr_u * 1) / (curr_point_idx + 2)

        # approx_grad = np.random.randn(n_dims, n_dims)
        return moving_average_approx_subgradient, moving_average_weights

    def predict(self, representation_d, features, labels):
        _, weight_vector = self.inner_algo(representation_d, features, labels)
        # TODO Add label recovery operation
        predictions = np.random.randn(features.shape[0])

        return predictions

    def get_params(self):
        return {"meta_algo_regul_param": self.meta_algo_regul_param, "inner_regul_param": self.inner_regul_param}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def mtl_mse_scorer(predictions, true_labels):
    n_tasks = len(true_labels)

    metric = 0
    for task_idx in range(n_tasks):
        c_metric = mean_squared_error(true_labels[task_idx], predictions[task_idx])
        metric = metric + c_metric
    metric = metric / n_tasks

    return metric

# TODO 1) Input: tr_tasks_tr_sets, val_tasks_tr_sets, val_tasks_val_sets
# TODO    Output: (D_η, c)
# TODO       a) Call the optimizer wrt D
# TODO       b) Call the optimizer wrt w
# TODO       c) Predict labels given a (D, w) pair
# TODO       d) Check score given a list of true and pred labels


# np.random.seed(2)
# vec = [0, 1, 2, 3, 4, 5]
# everything = [3] * len(vec)
# curr = 0
# moving_average = curr
# for i in range(len(vec)):
#     prev = curr
#     curr = np.random.randn()
#
#     everything[i] = curr
#     moving_average = (moving_average*i + curr*1) / (i+1)
#
# print(moving_average)
# print(np.mean(everything))
