import numpy as np
import scipy as sp
import warnings
import time
from scipy import sparse
from scipy.sparse.linalg import eigsh
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
        print('LTL | optimizing for inner param: %12f and outer param: %12f' % (self.inner_regul_param, self.meta_algo_regul_param))
        n_dims = self.data_info.n_dims

        cvx = False   # True, False

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
                weight_vector_ts = convex_solver_primal(features, labels, self.inner_regul_param, representation_d)

            predictions_ts.append(self.predict(weight_vector_ts, data.features_ts[test_task]))
        test_scores.append(mtl_scorer(predictions_ts, [data.labels_ts[i] for i in data.test_task_indexes], dataset=self.data_info.dataset))

        tt = time.time()
        printout = "T: %(task)3d | test score: %(ts_score)8.4f | time: %(time)7.2f" % \
                   {'task': -1, 'ts_score': float(np.mean(test_scores)), 'time': float(time.time() - tt)}
        self.logger.log_event(printout)
        for task_idx, task in enumerate(data.tr_task_indexes):
            prev_theta = curr_theta

            # TODO Try both curr_representation_d and representation_d
            if cvx is False:
                loss_subgradient, _, _ = inner_algo(self.data_info.n_dims, self.inner_regul_param,
                                                    curr_representation_d, data.features_tr[task], data.labels_tr[task], train_plot=0)
            else:
                with warnings.catch_warnings():
                    loss_subgradient = conex_solver_dual(data.features_tr[task], data.labels_tr[task], self.inner_regul_param, curr_representation_d)

            # Approximate the gradient
            g = data.features_tr[task].T @ loss_subgradient
            approx_grad = - 1 / (2 * self.inner_regul_param * data.features_tr[task].shape[0] ** 2) * np.outer(g, g)

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
                    weight_vector_ts = convex_solver_primal(features, labels, self.inner_regul_param, representation_d)

                predictions_ts.append(self.predict(weight_vector_ts, data.features_ts[test_task]))
            test_scores.append(mtl_scorer(predictions_ts, [data.labels_ts[i] for i in data.test_task_indexes], dataset=self.data_info.dataset))
            printout = "T: %(task)3d | test score: %(ts_score)8.4f | time: %(time)7.2f" % \
                       {'task': task, 'ts_score': float(np.mean(test_scores)), 'time': float(time.time() - tt)}

            if float(np.mean(test_scores)) < -0.2:
                print('breaking')
                break
            self.logger.log_event(printout)
        print(test_scores)
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
        #     val_scores.append(mtl_scorer([predictions_val], [data.labels_ts[val_task]]))

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


def mtl_scorer(predictions, true_labels, dataset=None):
    n_tasks = len(true_labels)

    metric = 0
    for task_idx in range(n_tasks):
        if dataset == 'movielens100k':
            non_zero_idx = np.nonzero(true_labels[task_idx])[0]
            c_metric = mean_absolute_error(true_labels[task_idx][non_zero_idx], predictions[task_idx][non_zero_idx])
        elif dataset == 'schools':
            c_metric = 100 * explained_variance_score(true_labels[task_idx], predictions[task_idx])
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

        representation_d = np.eye(n_dims)

        tt = time.time()
        self.representation_d = representation_d

        predictions_ts = []

        for test_task_idx, test_task in enumerate(data.test_task_indexes):
            cvx = False  # True, False
            features = data.features_tr[test_task]
            labels = data.labels_tr[test_task]

            if cvx is False:
                _, weight_vector, obj = inner_algo(self.data_info.n_dims, self.inner_regul_param, representation_d, features, labels)
            else:
                weight_vector = convex_solver_primal(features, labels, self.inner_regul_param, representation_d)

            predictions = self.predict(data.features_ts[test_task], weight_vector)
            predictions_ts.append(predictions)
        test_scores = mtl_scorer(predictions_ts, [data.labels_ts[i] for i in data.test_task_indexes], dataset=self.data_info.dataset)

        printout = "test score: %(ts_score)6.4f | time: %(time)7.2f" % \
                   {'ts_score': float(np.mean(test_scores)), 'time': float(time.time() - tt)}
        self.logger.log_event(printout)

        # TODO Optimization wrt w on val tasks, training points
        # val_scores = []
        # for val_task_idx, val_task in enumerate(data.val_task_indexes):
        #     predictions_val = self.predict(representation_d, data.features_ts[val_task], data.labels_ts[val_task])
        #     val_scores.append(mtl_scorer([predictions_val], [data.labels_ts[val_task]]))

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
            print('iter %5d | %10.5f' % (curr_point_idx, curr_obj))
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


def inner_algo_pure(n_dims, inner_regul_param, representation_d, features, labels, inner_algo_method='algo_w', train_plot=0):

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
    obj = []

    curr_obj = 10**10
    n_iter = 0
    subgradient_vector = np.zeros(total_n_points)
    while n_iter < 100 * total_n_points or n_iter < total_n_points:
        n_iter = n_iter + 1
        prev_obj = curr_obj
        prev_weight_vector = curr_weight_vector
        curr_point = np.random.choice(range(total_n_points), 1)[0]

        # Compute subgradient
        u = subgradient(labels[curr_point], features[curr_point], prev_weight_vector)
        subgradient_vector[curr_point] = u

        # Update
        step = 1 / (inner_regul_param * (n_iter + 1))
        full_subgrad = representation_d @ features[curr_point, :] * u + inner_regul_param * prev_weight_vector
        # full_subgrad = features[curr_point_idx, :] * u + inner_regul_param * representation_d_inv * prev_weight_vector
        curr_weight_vector = prev_weight_vector - step * full_subgrad

        moving_average_weights = (moving_average_weights * (n_iter - 1) + curr_weight_vector * 1) / n_iter

        curr_obj = absolute_loss(features, labels, moving_average_weights) + penalty(moving_average_weights)
        obj.append(curr_obj)
        # print('iter %8d | obj: %16.5f | step: %16.8f' % (n_iter, curr_obj, step))
        # if n_iter % (50 * total_n_points) == 0:
        #     print('iter %8d | obj: %16.5f | step: %16.8f' % (n_iter, curr_obj, step))
        conv = np.abs(prev_obj - curr_obj) / prev_obj
        if conv < 1e-8:
            # print('Conv tol reached: epoch %8d | obj: %10.5f | step: %16.8f' % (n_iter, curr_obj, step))
            break

    if train_plot == 1:
        plt.figure(999)
        plt.clf()
        plt.plot(obj)
        plt.pause(0.1)

    return subgradient_vector, moving_average_weights, obj


def convex_solver_primal(features, labels, regul_param, representation_d):
    fista = False

    if fista is False:
        import cvxpy as cp
        x = cp.Variable(features.shape[1])
        objective = cp.Minimize(cp.sum_entries(cp.abs(features * x - labels)) + (regul_param / 2) * cp.quad_form(x, np.linalg.pinv(representation_d)))

        prob = cp.Problem(objective)
        try:
            prob.solve()
        except Exception as e:
            print(e)
            prob.solve(solver='SCS')
        weight_vector = np.array(x.value).ravel()
    else:


    return weight_vector


def conex_solver_dual(features, labels, regul_param, representation_d):
    import cvxpy as cp
    loss_subgradient = np.zeros(features.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n_points = features.shape[0]
        a = cp.Variable(n_points)

        constraints = [cp.norm(a, "inf") <= 1]
        expr = cp.indicator(constraints)

        objective = cp.Minimize((1 / n_points) * np.reshape(labels, [1, n_points]) * a + expr +
                                (1 / (2 * regul_param * n_points ** 2)) * cp.norm(sp.linalg.sqrtm(representation_d) @ features.T * a) ** 2)

        prob = cp.Problem(objective)
        try:
            prob.solve()
            loss_subgradient = np.array(a.value).ravel()
        except Exception as e:
            print(e)
            try:
                prob.solve(solver='SCS')
            except Exception as e:
                print(e)
    return loss_subgradient


def fista(features, labels, regul_param, representation_d):
    if sparse.issparse(features) is False:
        n_points = features.shape[0]
    else:
        n_points = len(np.nonzero(labels)[0])

    largest_eigenval = eigsh(features @ features.T, k=1, which='LM')
    lipschitz_constant = (regul_param * n_points) / (np.linalg.norm(representation_d, ord="inf") * largest_eigenval)

    def penalty(xx):
        return 1 / (2 * regul_param * n_points) * np.linalg.norm(sp.linalg.sqrtm(representation_d) @ features.T @ xx, ord='fro')**2

    def prox(xx):
        # TODO Implement the prox
        return np.sign(xx) * np.maximum(abs(xx) - regul_param / lipschitz_constant, 0)

    def loss(xx):
        return 1 / n_points * labels @ xx

    def grad(xx):
        return 1 / (n_points ** 2 * regul_param) * features @ representation_d @ features.T @ xx

    pee = np.random.randn(n_points)
    alpha = np.random.randn(n_points)

    curr_iter = 0
    curr_cost = loss(alpha) + penalty(alpha)
    theta = 1
    objectives = []

    t = time.time()
    while curr_iter < 10 ** 5:
        curr_iter = curr_iter + 1
        prev_cost = curr_cost
        prev_pee = pee

        step_size = (1 / lipschitz_constant)
        search_point = alpha - step_size * grad(alpha)
        pee = prox(search_point)

        theta = (np.sqrt(theta ** 4 + 4 * theta ** 2) - theta ** 2) / 2
        rho = 1 - theta + np.sqrt(1 - theta)
        alpha = rho * pee - (rho - 1) * prev_pee

        curr_cost = loss(alpha) + penalty(alpha)

        objectives.append(curr_cost)
        diff = abs(prev_cost - curr_cost) / prev_cost

        if diff < 1e-5:
            break

        if time.time() - t > 0:
            t = time.time()
            print('iter: %6d | cost: %20.8f ~ tol: %18.15f | step: %12.10f' % (curr_iter, curr_cost, diff, step_size))
    return alpha