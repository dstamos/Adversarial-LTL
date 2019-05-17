import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, explained_variance_score


class LearningToLearnD:
    def __init__(self, data_info, logger, meta_algo_regul_param=0.1, inner_regul_param=0.1, verbose=1):
        self.verbose = verbose
        self.data_info = data_info
        self.logger = logger
        self.meta_algo_regul_param = meta_algo_regul_param
        self.inner_regul_param = inner_regul_param

        self.results = {'val_score': 0, 'test_scores': []}

        self.representation_d = None

    def fit(self, data):
        print('LTL | optimizing for inner param: %8e and outer param: %8e' % (self.inner_regul_param, self.meta_algo_regul_param))
        n_dims = self.data_info.n_dims

        curr_theta = np.zeros((n_dims, n_dims))
        representation_d = np.eye(n_dims) / n_dims

        test_scores = []
        predictions_ts = []
        for test_task_idx, test_task in enumerate(data.test_task_indexes):
            predictions_ts.append(self.predict(representation_d, data.features_ts[test_task], data.labels_ts[test_task]))
        test_scores.append(mtl_mae_scorer(predictions_ts, [data.labels_ts[i] for i in data.test_task_indexes]))

        tt = time.time()
        printout = "T: %(task)3d | test score: %(ts_score)8.4f | time: %(time)7.2f" % \
                   {'task': -1, 'ts_score': float(np.mean(test_scores)), 'time': float(time.time() - tt)}
        self.logger.log_event(printout)
        for task_idx, task in enumerate(data.tr_task_indexes):
            prev_theta = curr_theta

            loss_subgradient, _, _ = self.inner_algo(representation_d, data.features_tr[task], data.labels_tr[task], train_plot=0)

            # Approximate the gradient
            g = data.features_tr[task].T @ loss_subgradient
            approx_grad = - 1 / (2 * self.inner_regul_param * data.features_tr[task].shape[0] ** 2) * np.outer(g, g)

            # Update Theta
            curr_theta = prev_theta + approx_grad

            method = 'algo_b'
            if method == 'algo_a':
                # Compute M
                s, U = np.linalg.eig(-curr_theta/self.meta_algo_regul_param)
                U = np.real(U)
                s = np.real(s)
                s_exp = np.exp(s)

                matrix_m = U @ np.diag(s_exp) @ U.T

                # matrix_m = sp.linalg.expm(-curr_theta/self.meta_algo_regul_param)

                # Rescale
                curr_representation_d = matrix_m / np.trace(matrix_m)
            elif method == 'algo_b':
                curr_representation_d = - curr_theta/self.meta_algo_regul_param + np.eye(n_dims) / n_dims

                curr_representation_d = psd_trace_projection(curr_representation_d, 1)

            # Average:
            representation_d = (representation_d * (task_idx + 1) + curr_representation_d * 1) / (task_idx + 2)

            self.representation_d = representation_d

            predictions_ts = []
            for test_task_idx, test_task in enumerate(data.test_task_indexes):
                predictions_ts.append(self.predict(representation_d, data.features_ts[test_task], data.labels_ts[test_task]))
            test_scores.append(mtl_mae_scorer(predictions_ts, [data.labels_ts[i] for i in data.test_task_indexes]))

            printout = "T: %(task)3d | test score: %(ts_score)8.4f | time: %(time)7.2f" % \
                       {'task': task, 'ts_score': float(np.mean(test_scores)), 'time': float(time.time() - tt)}
            self.logger.log_event(printout)
        plt.figure(777)
        plt.clf()
        plt.plot(test_scores)
        plt.title('test scores ' + str(self.inner_regul_param) + ' | ' + str(self.meta_algo_regul_param))
        plt.annotate(str(test_scores[-1]), (self.data_info.n_test_tasks, test_scores[-1]))
        plt.pause(0.1)
        plt.savefig('schools-inner_' + str(self.inner_regul_param) + '-outer_' + str(self.meta_algo_regul_param) + '.png')

        # TODO Optimization wrt w on val tasks, training points
        val_scores = []
        for val_task_idx, val_task in enumerate(data.val_task_indexes):
            predictions_val = self.predict(representation_d, data.features_ts[val_task], data.labels_ts[val_task])
            val_scores.append(mtl_mae_scorer([predictions_val], [data.labels_ts[val_task]]))

        self.results['val_score'] = np.average(val_scores)
        self.results['test_scores'] = test_scores

    def inner_algo(self, representation_d, features, labels, inner_algo_method='algo_w', train_plot=0):

        representation_d_inv = np.linalg.pinv(representation_d)
        total_n_points = features.shape[0]

        if inner_algo_method == 'algo_w':
            def absolute_loss(curr_features, curr_labels, weight_vector):
                loss = np.linalg.norm(curr_labels - curr_features @ weight_vector, ord=1)
                loss = loss / total_n_points
                return loss

            def penalty(weight_vector):
                penalty_output = self.inner_regul_param / 2 * weight_vector @ representation_d_inv @ weight_vector
                # penalty_output = self.inner_regul_param / 2 * weight_vector @ np.linalg.lstsq(representation_d, weight_vector, rcond=None)[0]
                return penalty_output

            def subgradient(label, feature, weight_vector):
                pred = feature @ weight_vector
                subgrad = np.sign(pred - label)
                return subgrad

            def update_step(weight_vector, inner_iteration, subgrad, curr_epoch):
                step = 1 / (self.inner_regul_param * (curr_epoch*total_n_points + inner_iteration + 1 + 1))
                # TODO Change the pinv based on the computations in the paper
                full_subgrad = representation_d @ (features[inner_iteration, :] * subgrad + self.inner_regul_param * representation_d_inv @ weight_vector)
                new_weight_vector = weight_vector - step * full_subgrad
                return new_weight_vector
        else:
            raise ValueError("Unknown inner algorithm.")

        curr_weight_vector = np.zeros(self.data_info.n_dims)
        moving_average_weights = curr_weight_vector
        subgradient_vector = np.zeros(total_n_points)
        obj = []

        big_fucking_counter = 0
        for epoch in range(1):
            subgradient_vector = np.zeros(total_n_points)
            # TODO Shuffle points if we do multiple epochs?
            for curr_point_idx in range(features.shape[0]):
                big_fucking_counter = big_fucking_counter + 1
                prev_weight_vector = curr_weight_vector

                # Compute subgradient
                u = subgradient(labels[curr_point_idx], features[curr_point_idx], prev_weight_vector)
                subgradient_vector[curr_point_idx] = u

                # Update
                curr_weight_vector = update_step(prev_weight_vector, curr_point_idx, u, epoch)

                moving_average_weights = (moving_average_weights * (big_fucking_counter + 1) + curr_weight_vector * 1) / (big_fucking_counter + 2)

                obj.append(absolute_loss(features, labels, moving_average_weights) + penalty(moving_average_weights))
            conv = (obj[-2] - obj[-1]) / obj[-2]
            if conv < 10 ** -5:
                # print('breaking at %2d epochs' % epoch)
                break
                # print("online optimization | point: %4d | average loss: %9.5f" % (curr_point_idx, obj[-1]))
        # print(absolute_loss(features, labels, moving_average_weights))
        if train_plot == 1:
            plt.figure(999)
            plt.clf()
            plt.plot(obj)
            plt.pause(0.01)

        return subgradient_vector, moving_average_weights, absolute_loss(features, labels, moving_average_weights)

    def predict(self, representation_d, features, labels):
        _, weight_vector, _ = self.inner_algo(representation_d, features, labels)
        predictions = features @ weight_vector
        return predictions

    def get_params(self):
        return {"meta_algo_regul_param": self.meta_algo_regul_param, "inner_regul_param": self.inner_regul_param}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def mtl_mae_scorer(predictions, true_labels):
    n_tasks = len(true_labels)

    metric = 0
    for task_idx in range(n_tasks):
        c_metric = explained_variance_score(true_labels[task_idx], predictions[task_idx])
        # print(c_metric)
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


class IndipendentTaskLearning:
    def __init__(self, data_info, logger, meta_algo_regul_param=1e-1, inner_regul_param=0.1, verbose=1):
        self.verbose = verbose
        self.data_info = data_info
        self.logger = logger
        self.meta_algo_regul_param = meta_algo_regul_param
        self.inner_regul_param = inner_regul_param

        self.results = {'val_score': 0, 'test_scores': []}

        self.representation_d = None

    def fit(self, data):
        print('ITL | optimizing for inner param: %8e and outer param: %8e' % (self.inner_regul_param, self.meta_algo_regul_param))
        n_dims = data.data_info.n_dims

        test_scores = []

        representation_d = np.eye(n_dims) / n_dims

        tt = time.time()
        self.representation_d = representation_d

        predictions_ts = []
        for test_task_idx, test_task in enumerate(data.test_task_indexes):
            predictions_ts.append(self.predict(representation_d, data.features_ts[test_task], data.labels_ts[test_task]))
        test_scores.append(mtl_mae_scorer(predictions_ts, [data.labels_ts[i] for i in data.test_task_indexes]))

        printout = "test score: %(ts_score)6.4f | time: %(time)7.2f" % \
                   {'ts_score': float(np.mean(test_scores)), 'time': float(time.time() - tt)}
        self.logger.log_event(printout)

        # TODO Optimization wrt w on val tasks, training points
        val_scores = []
        for val_task_idx, val_task in enumerate(data.val_task_indexes):
            predictions_val = self.predict(representation_d, data.features_ts[val_task], data.labels_ts[val_task])
            val_scores.append(mtl_mae_scorer([predictions_val], [data.labels_ts[val_task]]))

        self.results['val_score'] = np.average(val_scores)
        self.results['test_scores'] = test_scores

    def inner_algo(self, representation_d, features, labels, inner_algo_method='algo_w', train_plot=0):

        representation_d_sqrt = sp.linalg.sqrtm(representation_d)
        representation_d_inv = np.linalg.pinv(representation_d)
        total_n_points = features.shape[0]

        if inner_algo_method == 'algo_v':
            def absolute_loss(curr_features, curr_labels, weight_vector):
                n_points = curr_features.shape[0]
                loss = np.sum(np.abs(curr_labels - curr_features @ representation_d_sqrt @ weight_vector))
                loss = loss / n_points
                return loss

            def penalty(weight_vector):
                penalty_output = self.inner_regul_param / 2 * np.linalg.norm(weight_vector, ord=2) ** 2
                return penalty_output

            def subgradient(label, feature, weight_vector):
                pred = feature @ representation_d_sqrt @ weight_vector
                subgrad_u = np.sign(label - pred) * (label - weight_vector)
                return subgrad_u

            def update_step(weight_vector, inner_iteration, epoch_idx):
                # FIXME curr_point_idx + i * len(points)
                new_weight_vector = weight_vector - 1 / (self.inner_regul_param * (epoch_idx*total_n_points + inner_iteration + 1)) * \
                                    (representation_d_sqrt @ features[inner_iteration, :] * u + self.inner_regul_param * weight_vector)
                return new_weight_vector
        elif inner_algo_method == 'algo_w':
            def absolute_loss(curr_features, curr_labels, weight_vector):
                loss = np.linalg.norm(curr_labels - curr_features @ weight_vector, ord=1)
                loss = loss / total_n_points
                return loss

            def penalty(weight_vector):
                penalty_output = self.inner_regul_param / 2 * weight_vector @ representation_d_inv @ weight_vector
                return penalty_output

            def subgradient(label, feature, weight_vector):
                pred = feature @ weight_vector
                subgrad = np.sign(pred - label)
                return subgrad

            def update_step(weight_vector, inner_iteration, subgrad, curr_epoch):
                step = 1 / (self.inner_regul_param * (curr_epoch*total_n_points + inner_iteration + 1))
                # TODO Change the pinv based on the computations in the paper
                full_subgrad = representation_d @ (features[inner_iteration, :] * subgrad + self.inner_regul_param * representation_d_inv @ weight_vector)
                new_weight_vector = weight_vector - step * full_subgrad
                return new_weight_vector
        else:
            raise ValueError("Unknown inner algorithm.")

        curr_weight_vector = np.zeros(self.data_info.n_dims)
        moving_average_weights = curr_weight_vector
        subgradient_vector = np.zeros(total_n_points)
        obj = []

        big_fucking_counter = 0
        for epoch in range(10):
            subgradient_vector = np.zeros(total_n_points)
            # TODO Shuffle points if we do multiple epochs?
            for curr_point_idx in range(features.shape[0]):
                big_fucking_counter = big_fucking_counter + 1
                prev_weight_vector = curr_weight_vector

                # Compute subgradient
                u = subgradient(labels[curr_point_idx], features[curr_point_idx], prev_weight_vector)
                subgradient_vector[curr_point_idx] = u

                # Update
                curr_weight_vector = update_step(prev_weight_vector, curr_point_idx, u, epoch)

                moving_average_weights = (moving_average_weights * (big_fucking_counter + 1) + curr_weight_vector * 1) / (big_fucking_counter + 2)

                obj.append(absolute_loss(features, labels, moving_average_weights) + penalty(moving_average_weights))
            conv = (obj[-2] - obj[-1]) / obj[-2]
            if conv < 10 ** -6:
                # print('breaking at %2d epochs' % epoch)
                break
                # print("online optimization | point: %4d | average loss: %9.5f" % (curr_point_idx, obj[-1]))
        # print(absolute_loss(features, labels, moving_average_weights))
        if train_plot == 1:
            plt.plot(obj)
            plt.pause(0.01)

        return subgradient_vector, moving_average_weights, absolute_loss(features, labels, moving_average_weights)

    def predict(self, representation_d, features, labels):
        _, weight_vector, _ = self.inner_algo(representation_d, features, labels)
        predictions = features @ weight_vector
        return predictions

    def get_params(self):
        return {"meta_algo_regul_param": self.meta_algo_regul_param, "inner_regul_param": self.inner_regul_param}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
