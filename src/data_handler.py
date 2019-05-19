import numpy as np
import scipy.io as sio
from numpy.linalg import norm
from sklearn.model_selection import train_test_split


class DataHandler:
    def __init__(self, data_info):
        self.data_info = data_info
        self.features_tr = [None] * data_info.n_all_tasks
        self.features_ts = [None] * data_info.n_all_tasks
        self.labels_tr = [None] * data_info.n_all_tasks
        self.labels_ts = [None] * data_info.n_all_tasks

        self.tr_task_indexes = None
        self.val_task_indexes = None
        self.test_task_indexes = None

        if self.data_info.dataset == 'synthetic':
            self.synthetic_data_gen()
        elif self.data_info.dataset == 'schools':
            self.schools_data_gen()

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
            weight_vector = (weight_vector / norm(weight_vector)).ravel()  # * np.random.randint(1, 10)

            # generating labels and adding noise
            clean_labels = features @ weight_vector
            noisy_labels = clean_labels + self.data_info.noise_std * np.random.randn(self.data_info.n_all_points)

            # split into training and test
            tr_indexes, ts_indexes = train_test_split(np.arange(0, self.data_info.n_all_points), test_size=self.data_info.ts_points_pct)
            features_tr = features[tr_indexes]
            labels_tr = noisy_labels[tr_indexes]

            features_ts = features[ts_indexes]
            # labels_ts = clean_labels[ts_indexes]
            labels_ts = noisy_labels[ts_indexes]

            self.features_tr[task_idx] = features_tr
            self.features_ts[task_idx] = features_ts
            self.labels_tr[task_idx] = labels_tr
            self.labels_ts[task_idx] = labels_ts

        # FIXME the random seed at the top of main then actually call the random seed at this point
        self.tr_task_indexes = np.arange(0, self.data_info.n_tr_tasks)
        self.val_task_indexes = np.arange(self.data_info.n_tr_tasks, self.data_info.n_tr_tasks + self.data_info.n_val_tasks)
        self.test_task_indexes = np.arange(self.data_info.n_tr_tasks + self.data_info.n_val_tasks, self.data_info.n_all_tasks)

    def schools_data_gen(self):

        temp = sio.loadmat('schoolData.mat')

        all_features = temp['X'][0]
        all_labels = temp['Y'][0]

        all_features = [all_features[i].T for i in range(len(all_features))]

        self.data_info.n_dims = all_features[0].shape[1]
        shuffled_task_indexes = np.random.permutation(self.data_info.n_all_tasks)

        for task_counter, task in enumerate(shuffled_task_indexes):
            # loading and normalizing the inputs
            features = all_features[task]
            features = features / norm(features, axis=1, keepdims=True)

            # loading the labels
            labels = all_labels[task].ravel() / 200

            n_points = len(labels)

            if task_counter >= self.data_info.n_tr_tasks:
                # split into training and test
                tr_indexes, ts_indexes = train_test_split(np.arange(0, n_points), test_size=self.data_info.ts_points_pct)
                features_tr = features[tr_indexes]
                labels_tr = labels[tr_indexes]

                features_ts = features[ts_indexes]
                labels_ts = labels[ts_indexes]

                self.features_tr[task] = features_tr
                self.features_ts[task] = features_ts
                self.labels_tr[task] = labels_tr
                self.labels_ts[task] = labels_ts
            else:
                self.features_tr[task] = features
                self.labels_tr[task] = labels

        # for task_counter, task in enumerate(shuffled_task_indexes):
        #     import sklearn
        #     scaler = sklearn.preprocessing.StandardScaler()
        #     scaler = scaler.fit(self.labels_tr[task_counter].reshape(-1, 1))
        #     self.labels_tr[task_counter] = scaler.transform(self.labels_tr[task_counter].reshape(-1, 1))
        #     try:
        #         scaler = scaler.fit(self.labels_ts[task_counter].reshape(-1, 1))
        #         self.labels_ts[task_counter] = scaler.transform(self.labels_ts[task_counter].reshape(-1, 1))
        #     except:
        #         pass
        #
        # max_value = 0
        # for task_counter, task in enumerate(shuffled_task_indexes):
        #     labels_max_1 = max(self.labels_tr[task_counter])
        #     try:
        #         labels_max_2 = max(self.labels_ts[task_counter])
        #         print(self.labels_ts[task_counter])
        #     except:
        #         labels_max_2 = 0
        #     if max(labels_max_1, labels_max_2) > max_value:
        #         max_value = max(labels_max_1, labels_max_2)
        #
        #
        # for task_counter, task in enumerate(shuffled_task_indexes):
        #     self.labels_tr[task_counter] = self.labels_tr[task_counter] / max_value
        #     try:
        #         self.labels_ts[task_counter] = self.labels_ts[task_counter] / max_value
        #     except:
        #         k = 1

        # FIXME the random seed at the top of main then actually call the random seed at this point
        self.tr_task_indexes = shuffled_task_indexes[:self.data_info.n_tr_tasks]
        self.val_task_indexes = shuffled_task_indexes[self.data_info.n_tr_tasks:self.data_info.n_tr_tasks + self.data_info.n_val_tasks]
        self.test_task_indexes = shuffled_task_indexes[self.data_info.n_tr_tasks + self.data_info.n_val_tasks:self.data_info.n_all_tasks]
