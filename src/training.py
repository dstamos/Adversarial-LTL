import numpy as np
import scipy as sp
import warnings
import time
from scipy import sparse
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, explained_variance_score


class LearningToLearnD:
    def __init__(self, data_info, logger, training_info, verbose=1):
        self.verbose = verbose
        self.data_info = data_info
        self.logger = logger
        self.meta_algo_regul_param = training_info.meta_algo_regul_param
        self.inner_regul_param = training_info.inner_regul_param
        self.training_info = training_info

        self.results = {'val_score': 0, 'test_scores': []}

        self.representation_d = None

    def fit(self, data):
        print(self.training_info.method + ' | inner param: %12f and outer param: %12f' % (self.inner_regul_param, self.meta_algo_regul_param))
        n_dims = self.data_info.n_dims

        curr_theta = np.zeros((n_dims, n_dims))
        if self.training_info.method == 'LTL_Oracle-SGD':
            curr_representation_d = data.oracle
        else:
            curr_representation_d = np.eye(n_dims) / n_dims
        representation_d = curr_representation_d

        tt = time.time()
        if self.training_info.method == 'LTL_ERM-ERM':
            cvx_val_ts = True
        else:
            cvx_val_ts = False

        if self.training_info.method == 'LTL_ERM-SGD' or self.training_info.method == 'LTL_ERM-ERM':
            cvx_tr = True
        else:
            cvx_tr = False

        test_scores = []
        predictions_ts = []
        for test_task_idx, test_task in enumerate(data.test_task_indexes):
            features = data.features_tr[test_task]
            labels = data.labels_tr[test_task]

            if cvx_val_ts is False:
                if self.data_info.dataset == 'miniwikipedia':
                    _, weight_vector_ts, _ = inner_algo_classification(self.data_info.n_dims, self.inner_regul_param, representation_d, features, labels, n_classes=4, train_plot=0)
                else:
                    _, weight_vector_ts, _ = inner_algo(self.data_info.n_dims, self.inner_regul_param, representation_d, features, labels, train_plot=0)
            else:
                if self.data_info.dataset == 'miniwikipedia':
                    raise ValueError("Not implemented.")
                else:
                    weight_vector_ts = convex_solver_primal(features, labels, self.inner_regul_param, representation_d)

            predictions_ts.append(self.predict(weight_vector_ts, data.features_ts[test_task]))
        if self.data_info.dataset == 'miniwikipedia':
            test_scores.append(mtl_scorer(predictions_ts, [data.labels_ts[i] for i in data.test_task_indexes], dataset=self.data_info.dataset, n_classes=4))
        else:
            test_scores.append(mtl_scorer(predictions_ts, [data.labels_ts[i] for i in data.test_task_indexes], dataset=self.data_info.dataset))

        printout = "T: %(task)3d | test score: %(ts_score)8.4f | time: %(time)7.2f" % {'task': -1, 'ts_score': float(np.mean(test_scores)), 'time': float(time.time() - tt)}
        self.logger.log_event(printout)
        hourglass = time.time()
        for task_idx, task in enumerate(data.tr_task_indexes):
            prev_theta = curr_theta

            if cvx_tr is False:
                if self.data_info.dataset == 'miniwikipedia':
                    loss_subgradient, _, _ = inner_algo_classification(self.data_info.n_dims, self.inner_regul_param, curr_representation_d, data.features_tr[task], data.labels_tr[task], n_classes=4, train_plot=0)
                else:
                    loss_subgradient, _, _ = inner_algo(self.data_info.n_dims, self.inner_regul_param, curr_representation_d, data.features_tr[task], data.labels_tr[task], train_plot=0)
            else:
                if self.data_info.dataset == 'miniwikipedia':
                    loss_subgradient = convex_solver_primal_classification(data.features_tr[task], data.labels_tr[task], self.inner_regul_param, curr_representation_d, n_classes=4)
                else:
                    loss_subgradient = convex_solver_dual(data.features_tr[task], data.labels_tr[task], self.inner_regul_param, curr_representation_d)

            ##########################################################################################
            ##########################################################################################
            # CVX/SGD Reconciliation
            # total_n_points = data.features_tr[task].shape[0]
            #
            # def multiclass_hinge_loss(curr_features, curr_labels, weight_matrix):
            #     pred_scores = curr_features @ weight_matrix
            #
            #     indicator_part = np.ones(pred_scores.shape)
            #     indicator_part[np.arange(pred_scores.shape[0]), curr_labels] = 0
            #
            #     true = pred_scores[np.arange(pred_scores.shape[0]), curr_labels].reshape(-1, 1)
            #     true = np.tile(true, (1, 4))
            #
            #     loss = np.max(indicator_part + pred_scores - true, axis=1)
            #
            #     loss = np.sum(loss) / total_n_points
            #
            #     return loss
            #
            # inner_regul_param = self.inner_regul_param
            #
            # def penalty(weight_matrix):
            #     penalty_output = inner_regul_param / 2 * np.trace(weight_matrix.T @ np.linalg.pinv(curr_representation_d) @ weight_matrix)
            #     return penalty_output
            #
            # sgd_obj = multiclass_hinge_loss(data.features_tr[task], data.labels_tr[task], sgd) + penalty(sgd)
            # cvx_obj = multiclass_hinge_loss(data.features_tr[task], data.labels_tr[task], cvx) + penalty(cvx)
            #
            # print('sgd obj: ', sgd_obj)
            # print('cvx obj: ', cvx_obj)
            #
            # d_inv = np.linalg.pinv(curr_representation_d)
            # gg = d_inv @ cvx @ cvx.T @ d_inv
            # cvx_approx_grad = - (self.inner_regul_param / 2) * gg
            #
            # gg = loss_subgradient @ loss_subgradient.T
            # sgd_approx_grad = - gg / (2 * self.inner_regul_param * data.features_tr[task].shape[0] ** 2)
            # print('sgd_approx_grad: ', np.linalg.norm(sgd_approx_grad, 'fro'))
            # print('cvx_approx_grad: ', np.linalg.norm(cvx_approx_grad, 'fro'))
            ##########################################################################################
            ##########################################################################################

            # Approximate the gradient
            if self.data_info.dataset == 'miniwikipedia' and self.training_info.method != 'LTL_SGD-SGD':
                # using the primal solution
                d_inv = np.linalg.pinv(curr_representation_d)
                gg = d_inv @ loss_subgradient @ loss_subgradient.T @ d_inv
                approx_grad = - (self.inner_regul_param / 2) * gg
            elif self.data_info.dataset == 'miniwikipedia':
                gg = loss_subgradient @ loss_subgradient.T
                approx_grad = - gg / (2 * self.inner_regul_param * data.features_tr[task].shape[0] ** 2)
            else:
                g = data.features_tr[task].T @ loss_subgradient
                gg = np.outer(g, g)
                approx_grad = - gg / (2 * self.inner_regul_param * data.features_tr[task].shape[0] ** 2)

            # Update Theta
            curr_theta = prev_theta + approx_grad

            method = 'algo_b'
            if method == 'algo_a':
                # Compute M
                s, matrix_u = np.linalg.eig(-curr_theta/self.meta_algo_regul_param)
                matrix_u = np.real(matrix_u)
                s = np.real(s)
                s_exp = np.exp(s)
                matrix_m = matrix_u @ np.diag(s_exp) @ matrix_u.T

                # Rescale
                curr_representation_d = matrix_m / np.trace(matrix_m)
            elif method == 'algo_b':
                curr_representation_d = - curr_theta/self.meta_algo_regul_param + np.eye(n_dims) / n_dims
                curr_representation_d = psd_trace_projection(curr_representation_d, 1)

            if self.training_info.method == 'LTL_Oracle-SGD':
                curr_representation_d = data.oracle
                representation_d = data.oracle
            else:
                # Average:
                representation_d = (representation_d * (task_idx + 1) + curr_representation_d * 1) / (task_idx + 2)

            self.representation_d = representation_d
            predictions_ts = []
            for test_task_idx, test_task in enumerate(data.test_task_indexes):
                features = data.features_tr[test_task]
                labels = data.labels_tr[test_task]

                if cvx_val_ts is False:
                    if self.data_info.dataset == 'miniwikipedia':
                        _, weight_vector_ts, _ = inner_algo_classification(self.data_info.n_dims, self.inner_regul_param, representation_d, features, labels, n_classes=4, train_plot=0)
                    else:
                        _, weight_vector_ts, _ = inner_algo(self.data_info.n_dims, self.inner_regul_param, representation_d, features, labels, train_plot=0)
                else:
                    if self.data_info.dataset == 'miniwikipedia':
                        weight_vector_ts = convex_solver_primal_classification(features, labels, self.inner_regul_param, representation_d, n_classes=4)
                    else:
                        weight_vector_ts = convex_solver_primal(features, labels, self.inner_regul_param, representation_d)

                predictions_ts.append(self.predict(weight_vector_ts, data.features_ts[test_task]))
            if self.data_info.dataset == 'miniwikipedia':
                test_scores.append(mtl_scorer(predictions_ts, [data.labels_ts[i] for i in data.test_task_indexes], dataset=self.data_info.dataset, n_classes=4))
            else:
                test_scores.append(mtl_scorer(predictions_ts, [data.labels_ts[i] for i in data.test_task_indexes], dataset=self.data_info.dataset))
            printout = "T: %(task)3d | test score: %(ts_score)8.4f | time: %(time)7.2f" % {'task': task_idx, 'ts_score': float(np.mean(test_scores)), 'time': float(time.time() - tt)}

            if time.time() - hourglass > 30:
                self.results['val_score'] = np.nan
                self.results['test_scores'] = test_scores
                self.logger.save(self.results)
                hourglass = time.time()

            self.logger.log_event(printout)
        # print(test_scores)
        # plt.figure(777)
        # plt.clf()
        # plt.plot(test_scores)
        # plt.title('test scores ' + str(self.inner_regul_param) + ' | ' + str(self.meta_algo_regul_param))
        # plt.annotate(str(test_scores[-1]), (self.data_info.n_test_tasks, test_scores[-1]))
        # plt.pause(0.1)
        # plt.savefig('schools-inner_' + str(self.inner_regul_param) + '-outer_' + str(self.meta_algo_regul_param) + '.png')

        predictions_val = []
        for val_task_idx, val_task in enumerate(data.val_task_indexes):
            features = data.features_tr[val_task]
            labels = data.labels_tr[val_task]

            if cvx_val_ts is False:
                if self.data_info.dataset == 'miniwikipedia':
                    _, weight_vector_val, _ = inner_algo_classification(self.data_info.n_dims, self.inner_regul_param, representation_d, features, labels, n_classes=4, train_plot=0)
                else:
                    _, weight_vector_val, _ = inner_algo(self.data_info.n_dims, self.inner_regul_param, representation_d, features, labels, train_plot=0)
            else:
                if self.data_info.dataset == 'miniwikipedia':
                    weight_vector_val = convex_solver_primal_classification(features, labels, self.inner_regul_param, representation_d, n_classes=4)
                else:
                    weight_vector_val = convex_solver_primal(features, labels, self.inner_regul_param, representation_d)

            predictions_val.append(self.predict(weight_vector_val, data.features_ts[val_task]))
        if self.data_info.dataset == 'miniwikipedia':
            val_score = mtl_scorer(predictions_val, [data.labels_ts[i] for i in data.val_task_indexes], dataset=self.data_info.dataset, n_classes=4)
        else:
            val_score = mtl_scorer(predictions_val, [data.labels_ts[i] for i in data.val_task_indexes], dataset=self.data_info.dataset)

        self.results['val_score'] = val_score
        self.results['test_scores'] = test_scores
        self.logger.save(self.results)

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


class IndipendentTaskLearning:
    def __init__(self, data_info, logger, training_info, verbose=1):
        self.verbose = verbose
        self.data_info = data_info
        self.logger = logger
        self.meta_algo_regul_param = training_info.meta_algo_regul_param
        self.inner_regul_param = training_info.inner_regul_param
        self.training_info = training_info

        self.results = {'val_score': 0, 'test_scores': []}

        self.representation_d = None

    def fit(self, data):
        print(self.training_info.method + ' | inner param: %8e and outer param: %8e' % (self.inner_regul_param, self.meta_algo_regul_param))
        n_dims = data.data_info.n_dims

        representation_d = np.eye(n_dims)

        self.representation_d = representation_d

        predictions_ts = []
        tt = time.time()
        for test_task_idx, test_task in enumerate(data.test_task_indexes):
            if self.training_info.method == 'ITL_ERM':
                cvx = True  # True, False
            else:
                cvx = False
            features = data.features_tr[test_task]
            labels = data.labels_tr[test_task]

            if cvx is False:
                if self.data_info.dataset == 'miniwikipedia':
                    _, weight_vector, obj = inner_algo_classification(self.data_info.n_dims, self.inner_regul_param, representation_d, features, labels, n_classes=4)
                else:
                    _, weight_vector, obj = inner_algo(self.data_info.n_dims, self.inner_regul_param, representation_d, features, labels)
            else:
                if self.data_info.dataset == 'miniwikipedia':
                    weight_vector = convex_solver_primal_classification(features, labels, self.inner_regul_param, representation_d, n_classes=4)
                else:
                    weight_vector = convex_solver_primal(features, labels, self.inner_regul_param, representation_d)

            predictions = self.predict(data.features_ts[test_task], weight_vector)

            predictions_ts.append(predictions)
            print('T: %3d/%3d trained | %7.2f' % (test_task_idx, len(data.test_task_indexes), time.time() - tt))
        if self.data_info.dataset == 'miniwikipedia':
            test_scores = mtl_scorer(predictions_ts, [data.labels_ts[i] for i in data.test_task_indexes], dataset=self.data_info.dataset, n_classes=4)
        else:
            test_scores = mtl_scorer(predictions_ts, [data.labels_ts[i] for i in data.test_task_indexes], dataset=self.data_info.dataset)

        printout = "test score: %(ts_score)6.4f | time: %(time)7.2f" % {'ts_score': float(np.mean(test_scores)), 'time': float(time.time() - tt)}
        self.logger.log_event(printout)

        self.results['val_score'] = test_scores
        self.results['test_scores'] = test_scores
        self.logger.save(self.results)

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


class AverageRating:
    def __init__(self, data_info, logger, training_info, verbose=1):
        self.verbose = verbose
        self.data_info = data_info
        self.logger = logger
        self.meta_algo_regul_param = training_info.meta_algo_regul_param
        self.inner_regul_param = training_info.inner_regul_param
        self.training_info = training_info

        self.results = {'val_score': 0, 'test_scores': []}

        self.representation_d = None

    def fit(self, data):
        print(self.training_info.method + ' | inner param: %8e and outer param: %8e' % (self.inner_regul_param, self.meta_algo_regul_param))
        n_dims = data.data_info.n_dims

        representation_d = np.eye(n_dims)

        self.representation_d = representation_d

        predictions_ts = []
        tt = time.time()
        for test_task_idx, test_task in enumerate(data.test_task_indexes):
            if self.training_info.method == 'ITL_ERM':
                cvx = True  # True, False
            else:
                cvx = False
            features = data.features_tr[test_task]
            labels = data.labels_tr[test_task]

            if cvx is False:
                _, weight_vector, obj = inner_algo(self.data_info.n_dims, self.inner_regul_param, representation_d, features, labels)
            else:
                weight_vector = convex_solver_primal(features, labels, self.inner_regul_param, representation_d)

            predictions = self.predict(data.features_ts[test_task], weight_vector)

            indeces_of_interest = np.nonzero(np.diag(data.features_ts[test_task].toarray()))[0]
            for idx_of_interest in indeces_of_interest:
                column_of_labels_of_interest = data.full_matrix[:, idx_of_interest].toarray()
                column_of_labels_of_interest[column_of_labels_of_interest == 0] = np.nan
                means = np.nanmean(column_of_labels_of_interest)
                predictions[idx_of_interest] = means

            predictions_ts.append(predictions)
            print('T: %3d/%3d trained | %7.2f' % (test_task_idx, len(data.test_task_indexes), time.time() - tt))
        test_scores = mtl_scorer([predictions_ts[i][np.nonzero(data.labels_ts[v])] for i, v in enumerate(data.test_task_indexes)],
                                 [data.labels_ts[i][np.nonzero(data.labels_ts[i])] for i in data.test_task_indexes], dataset=self.data_info.dataset)

        printout = "test score: %(ts_score)6.4f | time: %(time)7.2f" % {'ts_score': float(np.mean(test_scores)), 'time': float(time.time() - tt)}
        self.logger.log_event(printout)

        self.results['val_score'] = test_scores
        self.results['test_scores'] = test_scores
        self.logger.save(self.results)

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


def mtl_scorer(predictions, true_labels, dataset=None, n_classes=None):
    n_tasks = len(true_labels)

    metric = 0
    for task_idx in range(n_tasks):
        if dataset == 'movielens100k' or dataset == 'jester1':
            non_zero_idx = np.nonzero(true_labels[task_idx])[0]
            c_metric = mean_absolute_error(true_labels[task_idx][non_zero_idx], predictions[task_idx][non_zero_idx])
        elif dataset == 'schools':
            c_metric = 100 * explained_variance_score(true_labels[task_idx], predictions[task_idx])
        elif dataset == 'miniwikipedia':
            def multiclass_hinge_loss(curr_labels, pred_scores):
                indicator_part = np.ones(pred_scores.shape)
                indicator_part[np.arange(pred_scores.shape[0]), curr_labels] = 0

                true = pred_scores[np.arange(pred_scores.shape[0]), curr_labels].reshape(-1, 1)
                true = np.tile(true, (1, n_classes))

                loss = np.max(indicator_part + pred_scores - true, axis=1)

                loss = np.sum(loss) / len(curr_labels)

                return loss

            c_metric = multiclass_hinge_loss(true_labels[task_idx], predictions[task_idx])

            # from sklearn.metrics import hinge_loss
            # labels = np.array([0, 1, 2, 3])
            # c_metric = hinge_loss(true_labels[task_idx], predictions[task_idx], labels)
        else:
            c_metric = mean_absolute_error(true_labels[task_idx], predictions[task_idx])

        metric = metric + c_metric
    metric = metric / n_tasks

    return metric


def psd_trace_projection(matrix_d, constraint):

    s, matrix_u = np.linalg.eigh(matrix_d)
    s = np.maximum(s, 0)

    if np.sum(s) < constraint:
        return matrix_u @ np.diag(s) @ matrix_u.T

    search_points = np.insert(s, 0, 0)
    low_idx = 0
    high_idx = len(search_points)-1
    mid_idx = None
    matching_point = None
    s_sum = None

    def obj(vec, x):
        return np.sum(np.maximum(vec - x, 0))

    while low_idx <= high_idx:
        mid_idx = np.int(np.round((low_idx + high_idx) / 2))
        s_sum = obj(s, search_points[mid_idx])

        if s_sum == constraint:
            s = np.sort(s)
            d_proj = matrix_u @ np.diag(s) @ matrix_u.T
            return d_proj
        elif s_sum > constraint:
            low_idx = mid_idx + 1
        elif s_sum < constraint:
            high_idx = mid_idx - 1

    if s_sum > constraint:
        slope = (s_sum - obj(s, search_points[mid_idx+1])) / (search_points[mid_idx] - search_points[mid_idx+1])
        intercept = s_sum - slope * search_points[mid_idx]

        matching_point = (constraint - intercept) / slope
        # s_sum = obj(s, matching_point)
    elif s_sum < constraint:
        slope = (s_sum - obj(s, search_points[mid_idx-1])) / (search_points[mid_idx] - search_points[mid_idx-1])
        intercept = s_sum - slope * search_points[mid_idx]

        matching_point = (constraint - intercept) / slope
        # s_sum = obj(s, matching_point)

    s = np.maximum(s - matching_point, 0)
    s = np.sort(s)
    d_proj = matrix_u @ np.diag(s) @ matrix_u.T

    return d_proj


def inner_algo(n_dims, inner_regul_param, representation_d, features, labels, inner_algo_method='algo_w', train_plot=0):

    representation_d_inv = np.linalg.pinv(representation_d)

    total_n_points = features.shape[0]

    if inner_algo_method == 'algo_w':
        def absolute_loss(curr_features, curr_labels, weight_vector):
            loss = np.linalg.norm(curr_labels - curr_features @ weight_vector, ord=1)
            if sparse.issparse(features) is False:
                loss = loss / total_n_points
            else:
                loss = loss / len(np.nonzero(labels)[0])
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
    for epoch in range(1):
        prev_epoch_obj = curr_epoch_obj
        subgradient_vector = np.zeros(total_n_points)
        if sp.sparse.issparse(features) is False:
            shuffled_points = np.random.permutation(range(features.shape[0]))
        else:
            shuffled_points = np.random.permutation(np.nonzero(labels)[0])

        for curr_point_idx, curr_point in enumerate(shuffled_points):
            big_fucking_counter = big_fucking_counter + 1
            prev_weight_vector = curr_weight_vector

            # Compute subgradient
            u = subgradient(labels[curr_point], features[curr_point], prev_weight_vector)
            subgradient_vector[curr_point] = u

            # Update
            step = 1 / (inner_regul_param * (epoch * len(shuffled_points) + curr_point_idx + 1 + 1))
            if sp.sparse.issparse(features) is False:
                full_subgrad = representation_d @ features[curr_point, :] * u + inner_regul_param * prev_weight_vector
            else:
                full_subgrad = representation_d @ features[curr_point, :].toarray().ravel() * u + inner_regul_param * prev_weight_vector
            curr_weight_vector = prev_weight_vector - step * full_subgrad

            moving_average_weights = (moving_average_weights * (big_fucking_counter + 1) + curr_weight_vector * 1) / (big_fucking_counter + 2)

            curr_obj = absolute_loss(features, labels, curr_weight_vector) + penalty(curr_weight_vector)
            obj.append(curr_obj)
        # print('epoch %5d | obj: %10.5f | step: %16.10f' % (epoch, obj[-1], step))
        curr_epoch_obj = obj[-1]
        conv = np.abs(curr_epoch_obj - prev_epoch_obj) / prev_epoch_obj
        if conv < 1e-8:
            # print('BREAKING epoch %5d | obj: %10.5f | step: %16.10f' % (epoch, obj[-1], step))
            break

    if train_plot == 1:
        plt.figure(999)
        plt.clf()
        plt.plot(obj)
        plt.pause(0.1)

    return subgradient_vector, moving_average_weights, obj


def convex_solver_primal(features, labels, regul_param, representation_d):
    fista_method = True  # True, False

    if fista_method is False:
        import cvxpy as cp
        if sparse.issparse(features) is False:
            n_points = features.shape[0]
        else:
            n_points = len(np.nonzero(labels)[0])
        x = cp.Variable(features.shape[1])
        objective = cp.Minimize((1 / n_points) * cp.sum_entries(cp.abs(features * x - labels)) + (regul_param / 2) * cp.quad_form(x, np.linalg.pinv(representation_d)))

        prob = cp.Problem(objective)
        try:
            prob.solve()
        except Exception as e:
            print(e)
            prob.solve(solver='SCS')
        primal_weight_vector = np.array(x.value).ravel()
    else:
        primal_weight_vector, _ = fista(features, labels, regul_param, representation_d)

    # Sanity check
    # if sparse.issparse(features) is False:
    #     n_points = features.shape[0]
    # else:
    #     n_points = len(np.nonzero(labels)[0])
    #
    # ################################
    # # CVX
    # ################################
    # import cvxpy as cp
    # x = cp.Variable(features.shape[1])
    # objective = cp.Minimize((1 / n_points) * cp.sum_entries(cp.abs(features * x - labels)) + (regul_param / 2) * cp.quad_form(x, np.linalg.pinv(representation_d)))
    #
    # prob = cp.Problem(objective)
    # prob.solve()
    # primal_weight_vector = np.array(x.value).ravel()
    # print('cvx: \n', primal_weight_vector)
    # print('\n')
    # ################################
    # # Fista
    # ################################
    # primal_weight_vector, _ = fista(features, labels, regul_param, representation_d)
    # print('fista: \n', primal_weight_vector)
    # print('\n')
    return primal_weight_vector


def convex_solver_dual(features, labels, regul_param, representation_d):
    fista_method = True  # True, False

    if fista_method is False:
        import cvxpy as cp
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if sparse.issparse(features) is False:
                n_points = features.shape[0]
            else:
                n_points = len(np.nonzero(labels)[0])
            a = cp.Variable(features.shape[0])

            constraints = [cp.norm(a, "inf") <= 1]
            expr = cp.indicator(constraints)

            objective = cp.Minimize((1 / n_points) * np.reshape(labels, [1, features.shape[0]]) * a + expr +
                                    (1 / (2 * regul_param * n_points ** 2)) * cp.norm(sp.linalg.sqrtm(representation_d) @ features.T * a) ** 2)

            prob = cp.Problem(objective)
            try:
                prob.solve()
            except Exception as e:
                print(e)
                try:
                    prob.solve(solver='SCS')
                except Exception as e:
                    print(e)
            loss_subgradient = np.array(a.value).ravel()
    else:
        _, loss_subgradient = fista(features, labels, regul_param, representation_d)
    return loss_subgradient


def fista(features, labels, regul_param, representation_d):
    if sparse.issparse(features) is False:
        n_points = features.shape[0]
    else:
        n_points = len(np.nonzero(labels)[0])

    try:
        largest_eigenval = eigsh(representation_d, k=1, which='LM')[0][0]
    except Exception as e:
        print(e)
        largest_eigenval = sp.sparse.linalg.svds(representation_d)[1][-1]
    # lipschitz_constant = (largest_eigenval * max([np.linalg.norm(features[i, :]) for i in range(n_points)])) / (regul_param * n_points)
    lipschitz_constant = (largest_eigenval * 1) / (regul_param * n_points)
    step_size = (1 / lipschitz_constant)

    # def obj_fun(xx):
    #     return (1 / n_points) * np.linalg.norm(features @ xx - labels, ord=1) + \
    #            (regul_param / 2) * xx.T @ np.linalg.pinv(representation_d) @ xx

    def prox(xx):

        # Eq 120
        eta = step_size / n_points
        prox_diff = xx/eta - labels
        thresh = 1 / eta
        prox_point = np.copy(labels)
        prox_point[prox_diff < - thresh] = (xx/eta + thresh)[prox_diff < - thresh]
        prox_point[prox_diff > thresh] = (xx/eta - thresh)[prox_diff > thresh]

        # Moreau identity
        prox_point = xx - eta * prox_point
        return prox_point

    def grad(xx):
        return 1 / (n_points ** 2 * regul_param) * features @ representation_d @ features.T @ xx

    a = np.random.randn(features.shape[0])
    p = a
    primal_weight_vector = np.random.randn(features.shape[1])

    curr_iter = 0
    # curr_cost = obj_fun(primal_weight_vector)
    tau = 1
    # objectives = []

    t = time.time()
    while curr_iter < 10 ** 4:
        curr_iter = curr_iter + 1
        # prev_cost = curr_cost
        prev_tau = tau
        prev_a = a

        a = prox(p - step_size * grad(p))
        # print('iter: ', curr_iter)
        # print(a)
        # print('\n')

        primal_weight_vector = - 1 / (n_points * regul_param) * representation_d @ features.T @ a

        # theta = (np.sqrt(theta ** 4 + 4 * theta ** 2) - theta ** 2) / 2
        # rho = 1 - theta + np.sqrt(1 - theta)
        # p = rho * a - (rho - 1) * prev_a

        tau = (1 + np.sqrt(1 + 4 * prev_tau**2)) / 2
        p = a + ((prev_tau - 1) / tau) * (a - prev_a)

        diff = np.linalg.norm(a - prev_a) / np.linalg.norm(prev_a)
        # curr_cost = obj_fun(primal_weight_vector)
        # objectives.append(curr_cost)
        # diff = abs(prev_cost - curr_cost) / prev_cost

        # print('iter: ', curr_iter)
        # print(primal_weight_vector)
        # print('\n')
        if diff < 1e-5:
            break

        if time.time() - t > 30:
            t = time.time()
            print('iter: %6d | tol: %18.15f | step: %12.10f' % (curr_iter, diff, step_size))
    return primal_weight_vector, a


def inner_algo_classification(n_dims, inner_regul_param, representation_d, features, labels, inner_algo_method='algo_w', n_classes=None, train_plot=0):

    # representation_d_inv = np.linalg.pinv(representation_d + 1e-5 * np.eye(n_dims))
    representation_d_inv = np.linalg.pinv(representation_d)

    total_n_points = features.shape[0]

    if inner_algo_method == 'algo_w':
        def multiclass_hinge_loss(curr_features, curr_labels, weight_matrix):
            pred_scores = curr_features @ weight_matrix

            indicator_part = np.ones(pred_scores.shape)
            indicator_part[np.arange(pred_scores.shape[0]), curr_labels] = 0

            true = pred_scores[np.arange(pred_scores.shape[0]), curr_labels].reshape(-1, 1)
            true = np.tile(true, (1, 4))

            loss = np.max(indicator_part + pred_scores - true, axis=1)

            loss = np.sum(loss) / total_n_points

            return loss

        def penalty(weight_matrix):
            penalty_output = inner_regul_param / 2 * np.trace(weight_matrix.T @ representation_d_inv @ weight_matrix)
            return penalty_output

        def subgradient(label, feature, weight_matrix):

            pred_scores = feature @ weight_matrix

            indicator_part = np.ones(pred_scores.shape)
            indicator_part[label] = 0

            true = pred_scores[label]
            true = np.ones(weight_matrix.shape[1]) * true

            j_star = np.argmax(indicator_part + pred_scores - true)

            subgrad = np.zeros(weight_matrix.shape)

            if label != j_star:
                subgrad[:, label] = -feature
                subgrad[:, j_star] = feature

            return subgrad
    else:
        raise ValueError("Unknown inner algorithm.")

    curr_weight_matrix = np.zeros((n_dims, n_classes))
    moving_average_weights = curr_weight_matrix
    obj = []
    subgradients = []

    curr_epoch_obj = 10**10
    big_fucking_counter = 0
    n_epochs = 10
    for epoch in range(n_epochs):
        # subgradients = []
        prev_epoch_obj = curr_epoch_obj
        shuffled_points = np.random.permutation(range(features.shape[0]))

        for curr_point_idx, curr_point in enumerate(shuffled_points):
            big_fucking_counter = big_fucking_counter + 1
            prev_weight_matrix = curr_weight_matrix

            # Compute subgradient
            s = subgradient(labels[curr_point], features[curr_point], prev_weight_matrix)
            subgradients.append(s)
            # print(len(subgradients))

            # Update
            step = 1 / (inner_regul_param * (epoch * len(shuffled_points) + curr_point_idx + 1 + 1))
            # full_subgrad = s + inner_regul_param * representation_d_inv @ prev_weight_matrix
            # curr_weight_matrix = prev_weight_matrix - step * representation_d @ full_subgrad

            full_subgrad = representation_d @ s + inner_regul_param * prev_weight_matrix
            curr_weight_matrix = prev_weight_matrix - step * full_subgrad

            moving_average_weights = (moving_average_weights * (big_fucking_counter + 1) + curr_weight_matrix * 1) / (big_fucking_counter + 2)

            curr_obj = multiclass_hinge_loss(features, labels, moving_average_weights) + penalty(moving_average_weights)
            obj.append(curr_obj)
        # print('epoch %5d | obj: %10.5f | step: %16.10f' % (epoch, obj[-1], step))
        curr_epoch_obj = obj[-1]
        conv = np.abs(curr_epoch_obj - prev_epoch_obj) / prev_epoch_obj
        if conv < 1e-8:
            # print('BREAKING epoch %5d | obj: %10.5f | step: %16.10f' % (epoch, obj[-1], step))
            break

    if train_plot == 1:
        plt.figure(999)
        plt.clf()
        plt.plot(obj)
        plt.ylim(top=3, bottom=0)
        plt.pause(0.05)

    final_subgradient = np.sum(subgradients, axis=0) / n_epochs

    return final_subgradient, moving_average_weights, obj[-1]


def convex_solver_primal_classification(features, labels, regul_param, representation_d, n_classes=None):

    import cvxpy as cp
    pinv_d = np.linalg.pinv(representation_d)

    def multiclass_hinge_loss(cvx_variable, curr_features, curr_labels):
        # for loop
        n_points = curr_features.shape[0]
        margin_matrix = np.ones((n_points, n_classes))
        margin_matrix[np.arange(n_points), curr_labels] = 0

        pred_scores = curr_features @ cvx_variable

        loss = 0
        for i in range(n_points):
            margin = margin_matrix[i, :]
            all_scores = pred_scores[i, :]
            loss = loss + cp.max(margin + all_scores - pred_scores[i, curr_labels[i]])

        return loss

    n_points = features.shape[0]
    x = cp.Variable(shape=(features.shape[1], n_classes), name='primal')

    quad_loss = 0
    for t in range(n_classes):
        quad_loss = quad_loss + cp.quad_form(x[:, t], pinv_d)

    f = (1 / n_points) * multiclass_hinge_loss(x, features, labels) + (regul_param / 2) * quad_loss
    prob = cp.Problem(cp.Minimize(f))
    try:
        prob.solve(verbose=True)
    except Exception as e:
        print(e)
        prob.solve(solver='SCS')

    primal_weight_vector = np.array(x.value)

    return primal_weight_vector
