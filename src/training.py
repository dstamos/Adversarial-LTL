import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


class LearningToLearnD:
    def __init__(self, data_info, logger, meta_algo_regul_param=1e-1, inner_regul_param=100, verbose=1):
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

        total = []
        for task_idx, task in enumerate(data.tr_task_indexes):
            prev_theta = curr_theta

            curr_u, weight_vector, error = self.inner_algo(curr_representation_d, data.features_tr[task], data.labels_tr[task], train_plot=1)

            # Approximate the gradient
            # approx_grad = np.linalg.pinv(sp.linalg.sqrtm(curr_representation_d)) @ curr_u
            g = np.linalg.lstsq(sp.linalg.sqrtm(curr_representation_d), curr_u, rcond=-1)[0]
            approx_grad = - self.inner_regul_param / (2 * data.features_tr[task].shape[0]) * np.outer(g, g)

            # representation_d = representation_d - 1/self.meta_algo_regul_param * approx_grad
            #
            # representation_d = psd_trace_projection(representation_d, 1)

            # Update Theta
            curr_theta = prev_theta + approx_grad

            # Compute M
            matrix_m = sp.linalg.expm(-curr_theta/self.meta_algo_regul_param)

            # Rescale
            try:
                curr_representation_d = matrix_m / np.trace(matrix_m)
                print(curr_representation_d)
                print(np.all(np.isnan(curr_representation_d)))
            except:
                k = 1

            if np.all(np.isnan(curr_representation_d)) == True:
                k = 1

            # Average:
            representation_d = (representation_d * (task_idx + 1) + curr_representation_d * 1) / (task_idx + 2)
            self.representation_d = representation_d

            if np.all(np.isnan(representation_d)) == True:
                k = 1

            # total.append(error)
            # printout = "T: %(task)3d | train score: %(ts_score)6.4f" % {'task': task, 'ts_score': float(np.mean(total))}

            predictions_ts = []
            for test_task_idx, test_task in enumerate(data.test_task_indexes):
                # print("test task: %3d" % test_task)
                predictions_ts.append(self.predict(representation_d, data.features_ts[test_task], data.labels_ts[test_task]))
            test_scores.append(mtl_mse_scorer(predictions_ts, [data.labels_ts[i] for i in data.test_task_indexes]))

            printout = "T: %(task)3d | test score: %(ts_score)6.4f" % {'task': task, 'ts_score': float(np.mean(test_scores))}
            self.logger.log_event(printout)

        # TODO Optimization wrt w on val tasks, training points
        # Reuse the same function as Algo 1 or 2 but just return w
        # for task_idx, task in enumerate(data.val_task_indexes):
        #     _, weight_vector = self.inner_algo(self.representation_d, data.features_ts[task], data.labels_ts[task])

        # TODO Performance check of (D, w) on val tasks, validation points
        # Call MSE

    def inner_algo(self, representation_d, features, labels, inner_algo_method='algo_w', train_plot=0):
        def absolute_loss(curr_features, curr_labels, weight_vector, d_sqrt=None):
            n_points = curr_features.shape[0]
            if inner_algo_method == 'algo_v':
                loss = np.sum(np.abs(curr_labels - curr_features @ d_sqrt @ weight_vector))
            elif inner_algo_method == 'algo_w':
                loss = np.sum(np.abs(curr_labels - curr_features @ weight_vector))
            else:
                raise ValueError("Unknown inner algorithm.")
            return loss / n_points

        def penalty(weight_vector, d_sqrt=None):
            if inner_algo_method == 'algo_v':
                penalty_output = self.inner_regul_param / 2 * np.linalg.norm(weight_vector, ord=2) ** 2
            elif inner_algo_method == 'algo_w':
                penalty_output = self.inner_regul_param / 2 * weight_vector @ np.linalg.pinv(d_sqrt) @ weight_vector
            else:
                raise ValueError("Unknown inner algorithm.")
            return penalty_output

        curr_weight_vector = np.zeros(self.data_info.n_dims)
        moving_average_weights = curr_weight_vector
        obj = []

        representation_d_sqrt = sp.linalg.sqrtm(representation_d)
        representation_inv = np.linalg.pinv(representation_d)

        big_fucking_counter = 0
        for i in range(100):
            for curr_point_idx in range(features.shape[0]):
                big_fucking_counter = big_fucking_counter + 1
                prev_weight_vector = curr_weight_vector

                print(curr_weight_vector)

                # Compute subgradient
                true = labels[curr_point_idx]
                if inner_algo_method == 'algo_v':
                    pred = features[curr_point_idx, :] @ representation_d_sqrt @ prev_weight_vector
                elif inner_algo_method == 'algo_w':
                    pred = features[curr_point_idx, :] @ prev_weight_vector
                else:
                    raise ValueError("Unknown inner algorithm.")
                u = np.sign(true - pred) * (true - pred)

                # Update
                if inner_algo_method == 'algo_v':
                    curr_weight_vector = prev_weight_vector - 1 / (self.inner_regul_param * (curr_point_idx + i + 1)) * \
                                         (representation_d_sqrt @ features[curr_point_idx, :] * u + self.inner_regul_param * prev_weight_vector)
                elif inner_algo_method == 'algo_w':
                    curr_weight_vector = prev_weight_vector - 1 / (self.inner_regul_param * (curr_point_idx + i + 1)) * \
                                         representation_d @ (features[curr_point_idx, :] * u + self.inner_regul_param * representation_inv @ prev_weight_vector)
                else:
                    raise ValueError("Unknown inner algorithm.")

                # TODO recheck the computation of the moving average
                moving_average_weights = (moving_average_weights * (big_fucking_counter + 1) + curr_weight_vector * 1) / (big_fucking_counter + 2)

                obj.append(absolute_loss(features, labels, moving_average_weights, representation_d_sqrt) + penalty(moving_average_weights, representation_d_sqrt))
                # print("online optimization | point: %4d | average loss: %7.5f" % (curr_point_idx, obj[-1]))
        last_weights = curr_weight_vector
        if train_plot == 1:
            plt.plot(obj)
            plt.pause(0.01)
        return last_weights, moving_average_weights, obj[-1]

    def predict(self, representation_d, features, labels):
        try:
            _, weight_vector, _ = self.inner_algo(representation_d, features, labels)
        except:
            k = 1
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


def psd_trace_projection(matrix_d, constraint):

    s, U = np.linalg.eigh(matrix_d)
    s = np.maximum(s, 0)

    if np.sum(s) < constraint:
        return U @ np.diag(s) @ U.T

    search_points = np.insert(s, 0, 0)
    low_idx = 0
    high_idx = len(search_points)-1

    obj = lambda vec, x: np.sum(np.maximum(vec - x, 0))

    while low_idx <= high_idx:
        mid_idx = np.int(np.round((low_idx + high_idx) / 2))
        s_sum = obj(s, search_points[mid_idx])

        if s_sum == constraint:
            s = np.sort(s)
            d_proj = U @ np.diag(s) @ U.T
            return d_proj
        elif s_sum > constraint:
            low_idx = mid_idx + 1
        elif s_sum < constraint:
            high_idx = mid_idx - 1

    if s_sum > constraint:
        slope = (s_sum - obj(s, search_points[mid_idx+1])) / (search_points[mid_idx] - search_points[mid_idx+1])
        intercept = s_sum - slope * search_points[mid_idx]

        matching_point = (constraint - intercept) / slope
        s_sum = obj(s, matching_point)
    elif s_sum < constraint:
        slope = (s_sum - obj(s, search_points[mid_idx-1])) / (search_points[mid_idx] - search_points[mid_idx-1])
        intercept = s_sum - slope * search_points[mid_idx]

        matching_point = (constraint - intercept) / slope
        s_sum = obj(s, matching_point)

    s = np.maximum(s - matching_point, 0)
    s = np.sort(s)
    d_proj = U @ np.diag(s) @ U.T

    return d_proj

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
