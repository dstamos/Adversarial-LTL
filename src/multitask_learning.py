import numpy as np
import scipy as sp
import time
from src.training import mtl_mae_scorer


class MultitaskLearning:
    def __init__(self, data_info, logger, meta_algo_regul_param=1e-1, inner_regul_param=0.1, verbose=1):
        self.verbose = verbose
        self.data_info = data_info
        self.logger = logger
        self.meta_algo_regul_param = meta_algo_regul_param
        self.inner_regul_param = inner_regul_param

        self.results = {'val_score': 0, 'test_scores': []}

        self.representation_d = None

    def fit(self, data):
        print('MTL | optimizing for inner param: %8e and outer param: %8e' % (self.inner_regul_param, self.meta_algo_regul_param))
        n_dims = data.data_info.n_dims
        tt = time.time()
        representation_d = np.eye(n_dims) / n_dims
        self.representation_d = representation_d

        weights_matrix = np.zeros((n_dims, data.data_info.n_all_tasks))

        # representation_d = self._solve_wrt_d(representation_d, data.features_tr, data.labels_tr, data.test_task_indexes, self.inner_regul_param)
        weights_matrix = self.solve_wrt_w(representation_d, data.features_tr, data.labels_tr, weights_matrix, data.test_task_indexes)

        predictions_ts = []
        for test_task_idx, test_task in enumerate(data.test_task_indexes):
            predictions_ts.append(self.predict(data.features_ts[test_task], weights_matrix[:, test_task]))
        test_scores = mtl_mae_scorer(predictions_ts, [data.labels_ts[i] for i in data.test_task_indexes])

        self.results['val_score'] = 0
        self.results['test_scores'] = test_scores

        printout = "T: %(task)3d | test score: %(ts_score)8.4f | time: %(time)7.2f" % {'task': -1, 'ts_score': float(np.mean(test_scores)), 'time': float(time.time() - tt)}
        self.logger.log_event(printout)

    @staticmethod
    def solve_wrt_w(repr_d, features, labels, weights_matrix, task_range):
        for _, task_idx in enumerate(task_range):
            n_points = len(labels[task_idx])
            curr_w_pred = (repr_d @ features[task_idx].T @ np.linalg.solve(features[task_idx] @ repr_d @ features[task_idx].T + n_points * np.eye(n_points), labels[task_idx])).ravel()
            weights_matrix[:, task_idx] = curr_w_pred
        return weights_matrix

    def _solve_wrt_d(self, repr_d, features_tr, labels_tr, task_range, param1):
        def batch_objective(input_repr_d):
            return sum([features_tr[i].shape[0] * np.linalg.norm(sp.linalg.inv(features_tr[i] @ input_repr_d @ features_tr[i].T +
                                                                               features_tr[i].shape[0] * np.eye(features_tr[i].shape[0])) @
                                                                 labels_tr[i]) ** 2 for i in task_range])

        def batch_grad(input_repr_d):
            return self.batch_grad_func(input_repr_d, task_range, features_tr, labels_tr, 1)

        repr_d = np.eye(repr_d.shape[0])
        curr_obj = batch_objective(repr_d)

        objectives = []
        n_iter = 1999999
        curr_tol1 = 10 ** 10
        curr_tol2 = 10 ** 10
        conv_tol_obj = 10 ** -4
        c_iter = 0
        step_size = None

        t = time.time()
        while (c_iter < n_iter) and (curr_tol1 > conv_tol_obj):
            prev_repr_d = repr_d
            prev_obj = curr_obj

            step_size = 10 ** 18
            grad = batch_grad(prev_repr_d)

            temp_repr_d = self._psd_trace_projection(prev_repr_d - step_size * grad, 1 / param1)
            temp_obj = batch_objective(temp_repr_d)

            while temp_obj > (prev_obj + np.trace(grad.T @ (temp_repr_d - prev_repr_d)) + 1 / (2 * step_size) * np.linalg.norm(prev_repr_d - temp_repr_d, ord='fro') ** 2):
                step_size = 0.5 * step_size

                temp_repr_d = self._psd_trace_projection(prev_repr_d - step_size * grad, 1 / param1)
                temp_obj = batch_objective(temp_repr_d)

            repr_d = self._psd_trace_projection(prev_repr_d - step_size * grad, 1 / param1)

            curr_obj = batch_objective(repr_d)
            objectives.append(curr_obj)

            curr_tol1 = abs(curr_obj - prev_obj) / prev_obj
            curr_tol2 = np.linalg.norm(grad, 'fro')
            c_iter = c_iter + 1

            if curr_obj > 1.001 * prev_obj:
                print('fucked')

            if curr_tol1 < 10 ** -14:
                break

            if time.time() - t > 0:
                t = time.time()
                print("iter: %5d | obj: %12.8f | objtol: %10e | gradtol: %10e | step: %5.3e" % (c_iter, curr_obj, curr_tol1, curr_tol2, step_size))

        print("iter: %5d | obj: %12.8f | objtol: %10e | gradtol: %10e | step: %5.3e" % (c_iter, curr_obj, curr_tol1, curr_tol2, step_size))

        return repr_d

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

    @staticmethod
    def _psd_trace_projection(matrix_d, constraint):

        s_sum = None
        mid_idx = None
        matching_point = None

        s, mat_u = np.linalg.eigh(matrix_d)
        s = np.maximum(s, 0)

        if np.sum(s) < constraint:
            return mat_u @ np.diag(s) @ mat_u.T

        search_points = np.insert(s, 0, 0)
        low_idx = 0
        high_idx = len(search_points) - 1

        def obj(vec, x):
            return np.sum(np.maximum(vec - x, 0))

        while low_idx <= high_idx:
            mid_idx = np.int(np.round((low_idx + high_idx) / 2))
            s_sum = obj(s, search_points[mid_idx])

            if s_sum == constraint:
                s = np.sort(s)
                d_proj = mat_u @ np.diag(s) @ mat_u.T
                return d_proj
            elif s_sum > constraint:
                low_idx = mid_idx + 1
            elif s_sum < constraint:
                high_idx = mid_idx - 1

        if s_sum > constraint:
            slope = (s_sum - obj(s, search_points[mid_idx + 1])) / (search_points[mid_idx] - search_points[mid_idx + 1])
            intercept = s_sum - slope * search_points[mid_idx]

            matching_point = (constraint - intercept) / slope
            # s_sum = obj(s, matching_point)
        elif s_sum < constraint:
            slope = (s_sum - obj(s, search_points[mid_idx - 1])) / (search_points[mid_idx] - search_points[mid_idx - 1])
            intercept = s_sum - slope * search_points[mid_idx]

            matching_point = (constraint - intercept) / slope
            # s_sum = obj(s, matching_point)

        s = np.maximum(s - matching_point, 0)
        s = np.sort(s)
        d_proj = mat_u @ np.diag(s) @ mat_u.T

        return d_proj

    @staticmethod
    def batch_grad_func(repr_d, task_indeces, features_tr, labels_tr, switch):
        features = [features_tr[i] for i in task_indeces]
        labels = [labels_tr[i] for i in task_indeces]
        n_dims = features[0].shape[1]

        def mat_m(input_repr_d, n, t):
            return features[t] @ input_repr_d @ features[t].T + n * np.eye(n)

        grad = np.zeros((n_dims, n_dims))
        for idx, _ in enumerate(task_indeces):
            n_points = len(labels[idx])

            labels[idx] = np.reshape(labels[idx], [1, len(labels[idx])])
            inv_m = sp.linalg.inv(mat_m(repr_d, n_points, idx))
            curr_grad = features[idx].T @ inv_m @ ((labels[idx].T @ labels[idx]) @ inv_m + inv_m @ (labels[idx].T @ labels[idx])) @ inv_m @ features[idx]

            curr_grad = -n_points * curr_grad

            if switch == 1:
                grad = grad + curr_grad
            else:
                grad = grad + curr_grad
        return grad
