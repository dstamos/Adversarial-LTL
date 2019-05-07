import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import train_test_split


class DataHandler:
    def __init__(self, data_info):
        self.data_info = data_info
        self.features_tr = [None] * data_info.n_all_tasks
        self.features_val = [None] * data_info.n_all_tasks
        self.features_ts = [None] * data_info.n_all_tasks
        self.labels_tr = [None] * data_info.n_all_tasks
        self.labels_val = [None] * data_info.n_all_tasks
        self.labels_ts = [None] * data_info.n_all_tasks

        self.tr_task_indexes = None
        self.val_task_indexes = None
        self.test_task_indexes = None

        if self.data_info.dataset == 'synthetic':
            self.synthetic_data_gen()

    def synthetic_data_gen(self):
        sparsity = int(np.round(0.5 * self.data_info.n_dims))
        fixed_sparsity = np.random.choice(np.arange(0, self.data_info.n_dims), sparsity, replace=False)

        for task_idx in range(self.data_info.n_all_tasks):
            # generating and normalizing the inputs
            features = np.random.randn(self.data_info.n_all_points, self.data_info.n_dims)
            features = features / norm(features, axis=1, keepdims=True)

            # generating and normalizing the weight vectors
            weight_vector = np.zeros((self.data_info.n_dims, 1))
            weight_vector[fixed_sparsity] = np.random.randn(sparsity, 1)
            weight_vector = (weight_vector / norm(weight_vector)).ravel() * np.random.randint(1, 10)

            # generating labels and adding noise
            clean_labels = features @ weight_vector
            noisy_labels = self.data_info.noise_std * np.random.randn(self.data_info.n_all_points)

            # split into training and test
            tr_val_indexes, ts_indexes = train_test_split(np.arange(0, self.data_info.n_all_points), test_size=1 - self.data_info.ts_points_pct)
            if task_idx < self.data_info.n_tr_tasks:
                features_tr = features[tr_val_indexes]
                labels_tr = noisy_labels[tr_val_indexes]
            else:
                tr_indexes, val_indexes = train_test_split(tr_val_indexes, test_size=self.data_info.val_points_pct)
                features_tr = features[tr_indexes]
                labels_tr = noisy_labels[tr_indexes]
                features_val = features[val_indexes]
                labels_val = noisy_labels[val_indexes]
                self.features_val[task_idx] = features_val
                self.labels_val[task_idx] = labels_val

            features_ts = features[ts_indexes]
            labels_ts = clean_labels[ts_indexes]

            self.features_tr[task_idx] = features_tr
            self.features_ts[task_idx] = features_ts
            self.labels_tr[task_idx] = labels_tr
            self.labels_ts[task_idx] = labels_ts

        # FIXME the random seed at the top of main then actually call the random seed at this point
        self.tr_task_indexes = np.arange(0, self.data_info.n_tr_tasks)
        self.val_task_indexes = np.arange(self.data_info.n_tr_tasks, self.data_info.n_tr_tasks + self.data_info.n_val_tasks)
        self.test_task_indexes = np.arange(self.data_info.n_tr_tasks + self.data_info.n_val_tasks, self.data_info.n_all_tasks)
