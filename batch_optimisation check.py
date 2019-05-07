import cvxpy as cvx
import numpy as np
import matplotlib.pylab as plt
import cvxpy as cvx

import scs

# Ensure repeatably random problem data.
np.random.seed(0)


# n_dims = data_settings['n_dims']
n_dims = D.shape[0]
n_tasks = len(task_range)


W = np.random.randn(n_dims, n_tasks)


W = cvx.Variable(n_dims)
objective = cvx.Minimize(
            (1/n_tasks) * cvx.square(cvx.norm(Y_train[i] - X_train[i] * W)) + param1 * cvx.matrix_frac(W, D)
            )
            # sum([
            #     (1/n_tasks) * cvx.square(cvx.norm(Y_train[i] - X_train[i] * W[:,i])) + param1 * cvx.matrix_frac(W[:, i], D)
            #     for i in [80]])
            # )
prob = cvx.Problem(objective)
prob.solve()
print(W.value)






# objective = cvx.Minimize(cvx.norm((X_train[i] @ D @ X_train[i].T + n_points[i] * eye(n_points[i]))*x - Y_train[i]))
objective = cvx.Minimize(cvx.norm(sp.linalg.inv(X_train[i] @ D1 @ X_train[i].T + n_points[i] * eye(n_points[i])) @ Y_train[i], 'fro'))
# cvx.matrix_frac(X_train[i].T, D1) + n_points[i] * eye(n_points[i])
prob = cvx.Problem(objective)
prob.solve()
print(x.value)























D = cvx.Variable(n_dims, n_dims)
obj = cvx.Minimize(sum([n_points[i] * cvx.square(cvx.norm(sp.linalg.inv(X_train[i] * D * X_train[i].T + n_points[i] * eye(n_points[i])) @ Y_train[i], 'fro')) for i in task_range]))


sum([n_points[i] * cvx.norm(lstsq(X_train[i] * D * X_train[i].T + n_points[i] * eye(n_points[i]), Y_train[i], rcond=None)[0]) ** 2 for i in task_range])


# a subproblem for a specific task
x = cvx.Variable(n_points[i])
# objective = cvx.Minimize(cvx.norm((X_train[i] @ D @ X_train[i].T + n_points[i] * eye(n_points[i]))*x - Y_train[i]))
objective = cvx.Minimize(cvx.norm(sp.linalg.inv(X_train[i] @ D1 @ X_train[i].T + n_points[i] * eye(n_points[i])) @ Y_train[i], 'fro'))
# cvx.matrix_frac(X_train[i].T, D1) + n_points[i] * eye(n_points[i])
prob = cvx.Problem(objective)
prob.solve()
print(x.value)




batch_grad = lambda D: batch_grad_func(D, task_range, data, 1)






cvx.inv_pos(np.random.randn(3, 3))




# Perform alternating minimization.
max_iters = 30
residual = np.zeros(max_iters)
for iter_num in range(max_iters):
    # At the beginning of an iteration, X and Y are NumPy
    # array types, NOT CVXPY variables.

    # For odd iterations, treat Y constant, optimize over X.
    if iter_num % 2 == 1:
        X = cvx.Variable(k, n)
        constraint = [X >= 0]
    # For even iterations, treat X constant, optimize over Y.
    else:
        Y = cvx.Variable(m, k)
        constraint = [Y >= 0]

    # Solve the problem.
    obj = cvx.Minimize(cvx.norm(A - Y * X, 'fro'))
    prob = cvx.Problem(obj, constraint)
    prob.solve(solver=cvx.SCS)

    # if prob.status != cvx.OPTIMAL:
    #     raise Exception("Solver did not converge!")

    print('Iteration %3d, residual norm %7.5f' % (iter_num, prob.value))

    residual[iter_num-1] = prob.value

    # Convert variable to NumPy array constant for next iteration.
    if iter_num % 2 == 1:
        X = X.value
    else:
        Y = Y.value


plt.plot(residual)

plt.pause(0.01)
k=1