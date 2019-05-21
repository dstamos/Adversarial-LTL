import numpy as np
import scipy as sp
import cvxpy as cp
import warnings
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, explained_variance_score, mean_squared_error


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

        cvx = True   # True, False

        curr_theta = np.zeros((n_dims, n_dims))
        curr_representation_d = np.eye(n_dims) / n_dims
        representation_d = curr_representation_d

        test_scores = []
        predictions_ts = []
        for test_task_idx, test_task in enumerate(data.test_task_indexes):
            features = data.features_tr[test_task]
            labels = data.labels_tr[test_task]

            if cvx is False:
                _, weight_vector_ts, _ = inner_algo(self.data_info.n_dims, self.inner_regul_param, representation_d, features, labels, train_plot=0)
            else:
                x = cp.Variable(features.shape[1])
                objective = cp.Minimize(cp.sum_entries(cp.abs(features * x - labels)) + (self.inner_regul_param / 2) * cp.quad_form(x, np.linalg.pinv(representation_d)))

                prob = cp.Problem(objective)
                prob.solve()
                weight_vector_ts = np.array(x.value).ravel()

            predictions_ts.append(self.predict(weight_vector_ts, data.features_ts[test_task]))
        test_scores.append(mtl_mae_scorer(predictions_ts, [data.labels_ts[i] for i in data.test_task_indexes]))

        tt = time.time()
        printout = "T: %(task)3d | test score: %(ts_score)8.4f | time: %(time)7.2f" % \
                   {'task': -1, 'ts_score': float(np.mean(test_scores)), 'time': float(time.time() - tt)}
        self.logger.log_event(printout)
        for task_idx, task in enumerate(data.tr_task_indexes):
            prev_theta = curr_theta

            if cvx is False:
                loss_subgradient, _, _ = inner_algo(self.data_info.n_dims, self.inner_regul_param,
                                                    representation_d, data.features_tr[task], data.labels_tr[task], train_plot=0)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    features = data.features_tr[task]
                    labels = data.labels_tr[task]
                    n_points = features.shape[0]
                    a = cp.Variable(n_points)

                    constraints = [cp.norm(a, "inf") <= 1]
                    expr = cp.indicator(constraints)

                    objective = cp.Minimize((1/n_points) * np.reshape(labels, [1, n_points])*a + expr +
                                            (1 / (2 * self.inner_regul_param * n_points**2)) * cp.norm(sp.linalg.sqrtm(representation_d) @ features.T * a)**2)

                    prob = cp.Problem(objective)
                    prob.solve()
                    loss_subgradient = np.array(a.value).ravel()
            ###################################################################################################################
            # loss_subgradient_ours, _, _ = inner_algo(self.data_info.n_dims, self.inner_regul_param,
            #                                          representation_d, data.features_tr[task], data.labels_tr[task], train_plot=0)
            # features = data.features_tr[task]
            # labels = data.labels_tr[task]
            # n_points = features.shape[0]
            # a = cp.Variable(n_points)
            #
            # constraints = [cp.norm(a, "inf") <= 1]
            # expr = cp.indicator(constraints)
            #
            # objective = cp.Minimize((1/n_points) * np.reshape(labels, [1, n_points])*a + expr +
            #                         (1 / (2 * self.inner_regul_param * n_points**2)) * cp.norm(sp.linalg.sqrtm(representation_d) @ features.T * a)**2)
            #
            # prob = cp.Problem(objective)
            # prob.solve()
            # loss_subgradient_cvx = np.array(a.value).ravel()
            #
            # print('ours:')
            # print(loss_subgradient_ours)
            # print('cvx:')
            # print(loss_subgradient_cvx)
            ###################################################################################################################
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
                features = data.features_tr[test_task]
                labels = data.labels_tr[test_task]

                if cvx is False:
                    _, weight_vector_ts, _ = inner_algo(self.data_info.n_dims, self.inner_regul_param, representation_d, features, labels, train_plot=0)
                else:
                    x = cp.Variable(features.shape[1])
                    objective = cp.Minimize(cp.sum_entries(cp.abs(features * x - labels)) + (self.inner_regul_param / 2) * cp.quad_form(x, np.linalg.pinv(representation_d)))

                    prob = cp.Problem(objective)

                    prob.solve()
                    weight_vector_ts = np.array(x.value).ravel()

                predictions_ts.append(self.predict(weight_vector_ts, data.features_ts[test_task]))
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
        # val_scores = []
        # for val_task_idx, val_task in enumerate(data.val_task_indexes):
        #     predictions_val = self.predict(representation_d, data.features_ts[val_task], data.labels_ts[val_task])
        #     val_scores.append(mtl_mae_scorer([predictions_val], [data.labels_ts[val_task]]))

        self.results['val_score'] = 0   # np.average(val_scores)
        self.results['test_scores'] = test_scores

    @staticmethod
    def predict(weight_vector, features):
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
        # c_metric = 100 * explained_variance_score(true_labels[task_idx], predictions[task_idx])
        c_metric = mean_absolute_error(true_labels[task_idx], predictions[task_idx])

        # n_points = len(true_labels[task_idx])
        # mse = np.linalg.norm(true_labels[task_idx] - predictions[task_idx])**2 / n_points
        # c_metric = (1 - mse/np.var(true_labels[task_idx]))

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

        representation_d = np.eye(n_dims) / n_dims

        tt = time.time()
        self.representation_d = representation_d

        predictions_ts = []

        representation_d_inv = np.linalg.pinv(representation_d)

        for test_task_idx, test_task in enumerate(data.test_task_indexes):
            def absolute_loss(curr_features, curr_labels, weight_vector):
                loss = np.linalg.norm(curr_labels - curr_features @ weight_vector, ord=1)
                loss = loss / curr_features.shape[0]
                return loss

            def penalty(vectorsss):
                penalty_output = self.inner_regul_param / 2 * vectorsss @ representation_d_inv @ vectorsss
                return penalty_output

            # _, weight_vector, last_obj = inner_algo(self.data_info.n_dims, self.inner_regul_param, representation_d, data.features_tr[test_task], data.labels_tr[test_task])
            cvx = False
            features = data.features_tr[test_task]
            labels = data.labels_tr[test_task]

            if cvx is False:
                _, weight_vector, _ = inner_algo(self.data_info.n_dims, self.inner_regul_param, representation_d, features, labels)
            else:
                x = cp.Variable(features.shape[1])
                objective = cp.Minimize(cp.sum_entries(cp.abs(features * x - labels)) + (self.inner_regul_param / 2) * cp.quad_form(x, np.linalg.pinv(representation_d)))

                prob = cp.Problem(objective)

                prob.solve()
                weight_vector = np.array(x.value).ravel()

            # obj_ours = absolute_loss(data.features_tr[test_task], data.labels_tr[test_task], weight_vector) + penalty(weight_vector)
            #
            # import cvxpy as cp
            # x = cp.Variable(n_dims)
            # objective = cp.Minimize(cp.sum_entries(cp.abs(data.features_tr[test_task] * x - data.labels_tr[test_task])) +
            #                         (self.inner_regul_param / 2) * cp.quad_form(x, representation_d_inv))
            #
            # prob = cp.Problem(objective)
            #
            # result = prob.solve()
            # weight_vector = np.array(x.value).ravel()
            # print(weight_vector_cvx)
            #
            # obj_cvx = absolute_loss(data.features_tr[test_task], data.labels_tr[test_task], weight_vector_cvx) + penalty(weight_vector_cvx)
            #
            #
            #
            #
            # print('ours: %f' % obj_ours)
            # print('cvx: %f' % obj_cvx)

            predictions = self.predict(data.features_ts[test_task], weight_vector)
            predictions_ts.append(predictions)
        test_scores = mtl_mae_scorer(predictions_ts, [data.labels_ts[i] for i in data.test_task_indexes])

        printout = "test score: %(ts_score)6.4f | time: %(time)7.2f" % \
                   {'ts_score': float(np.mean(test_scores)), 'time': float(time.time() - tt)}
        self.logger.log_event(printout)

        # TODO Optimization wrt w on val tasks, training points
        # val_scores = []
        # for val_task_idx, val_task in enumerate(data.val_task_indexes):
        #     predictions_val = self.predict(representation_d, data.features_ts[val_task], data.labels_ts[val_task])
        #     val_scores.append(mtl_mae_scorer([predictions_val], [data.labels_ts[val_task]]))

        self.results['val_score'] = 0  # np.average(val_scores)
        self.results['test_scores'] = test_scores

    @staticmethod
    def predict(features, weight_vector):
        predictions = features @ weight_vector
        return predictions

    def get_params(self):
        return {"meta_algo_regul_param": self.meta_algo_regul_param, "inner_regul_param": self.inner_regul_param}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def inner_algo(n_dims, inner_regul_param, representation_d, features, labels, inner_algo_method='algo_w', train_plot=0):

    representation_d_inv = np.linalg.pinv(representation_d)
    total_n_points = features.shape[0]

    if inner_algo_method == 'algo_w':
        def absolute_loss(curr_features, curr_labels, weight_vector):
            loss = np.linalg.norm(curr_labels - curr_features @ weight_vector, ord=1)
            loss = loss / total_n_points
            return loss

        def penalty(weight_vector):
            penalty_output = inner_regul_param / 2 * weight_vector @ representation_d_inv @ weight_vector
            return penalty_output

        def subgradient(label, feature, weight_vector):
            pred = feature @ weight_vector
            subgrad = np.sign(pred - label)
            return subgrad

    else:
        raise ValueError("Unknown inner algorithm.")

    curr_weight_vector = np.zeros(n_dims)
    moving_average_weights = curr_weight_vector
    subgradient_vector = np.zeros(total_n_points)
    obj = []

    curr_epoch_obj = 10**10
    big_fucking_counter = 0
    for epoch in range(1000):
        prev_epoch_obj = curr_epoch_obj
        subgradient_vector = np.zeros(total_n_points)
        # TODO Shuffle points if we do multiple epochs?
        shuffled_points = np.random.permutation(range(features.shape[0]))
        for curr_point_idx, curr_point in enumerate(shuffled_points):
            big_fucking_counter = big_fucking_counter + 1
            prev_weight_vector = curr_weight_vector

            # Compute subgradient
            u = subgradient(labels[curr_point], features[curr_point], prev_weight_vector)
            subgradient_vector[curr_point_idx] = u

            # Update
            step = 1 / (inner_regul_param * (epoch * total_n_points + curr_point_idx + 1 + 1))
            full_subgrad = representation_d @ features[curr_point_idx, :] * u + inner_regul_param * prev_weight_vector
            # full_subgrad = features[curr_point_idx, :] * u + inner_regul_param * representation_d_inv * prev_weight_vector
            curr_weight_vector = prev_weight_vector - step * full_subgrad

            moving_average_weights = (moving_average_weights * (big_fucking_counter + 1) + curr_weight_vector * 1) / (big_fucking_counter + 2)

            obj.append(absolute_loss(features, labels, curr_weight_vector) + penalty(curr_weight_vector))
        # print('epoch %5d | obj: %10.5f | step: %16.10f' % (epoch, obj[-1], step))
        curr_epoch_obj = obj[-1]
        conv = np.abs(curr_epoch_obj - prev_epoch_obj) / prev_epoch_obj
        if conv < 1e-10:
            break

    if train_plot == 1:
        plt.figure(999)
        plt.clf()
        plt.plot(obj)
        plt.pause(0.1)

    return subgradient_vector, moving_average_weights, obj[-1]

