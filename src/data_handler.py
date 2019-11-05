import numpy as np
import scipy.io as sio
from scipy.sparse import csc_matrix
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from copy import deepcopy
import scipy as sp
import pickle
from sklearn.preprocessing import normalize
from operator import itemgetter
import os


class DataHandler:
    def __init__(self, data_info):
        self.data_info = data_info
        self.features_tr = [None] * data_info.n_all_tasks
        self.features_ts = [None] * data_info.n_all_tasks
        self.labels_tr = [None] * data_info.n_all_tasks
        self.labels_ts = [None] * data_info.n_all_tasks
        self.oracle = None

        self.tr_task_indexes = None
        self.val_task_indexes = None
        self.test_task_indexes = None

        if self.data_info.dataset == 'synthetic':
            self.synthetic_data_gen()
        elif self.data_info.dataset == 'synthetic_data_gen_biased_sgd_paper':
            self.synthetic_data_gen_biased_sgd_paper()
        elif self.data_info.dataset == 'schools':
            self.schools_data_gen()
        elif self.data_info.dataset == 'movielens100k':
            self.full_matrix = None
            self.movielens_gen()
        elif self.data_info.dataset == 'jester1':
            self.full_matrix = None
            self.jester_gen()
        elif self.data_info.dataset == 'miniwikipedia':
            self.miniwikipedia_gen()

    def synthetic_data_gen(self):
        sparsity = int(np.round(0.2 * self.data_info.n_dims))
        fixed_sparsity = np.random.choice(np.arange(0, self.data_info.n_dims), sparsity, replace=False)

        matrix_w = np.zeros((self.data_info.n_dims, self.data_info.n_all_tasks))
        for task_idx in range(self.data_info.n_all_tasks):
            # generating and normalizing the inputs
            features = np.random.randn(self.data_info.n_all_points, self.data_info.n_dims)
            features = features / norm(features, axis=1, keepdims=True)

            # generating and normalizing the weight vectors
            weight_vector = np.zeros((self.data_info.n_dims, 1))
            weight_vector[fixed_sparsity] = np.random.randn(sparsity, 1)
            weight_vector = (weight_vector / norm(weight_vector)).ravel()  # * np.random.randint(1, 10)

            matrix_w[:, task_idx] = weight_vector

            # generating labels and adding noise
            clean_labels = features @ weight_vector
            noisy_labels = clean_labels + self.data_info.noise_std * np.random.randn(self.data_info.n_all_points)

            # split into training and test
            tr_indexes, ts_indexes = train_test_split(np.arange(0, self.data_info.n_all_points), test_size=self.data_info.ts_points_pct)
            features_tr = features[tr_indexes]
            labels_tr = noisy_labels[tr_indexes]

            features_ts = features[ts_indexes]
            labels_ts = clean_labels[ts_indexes]
            # labels_ts = noisy_labels[ts_indexes]

            self.features_tr[task_idx] = features_tr
            self.features_ts[task_idx] = features_ts
            self.labels_tr[task_idx] = labels_tr
            self.labels_ts[task_idx] = labels_ts

        self.tr_task_indexes = np.arange(0, self.data_info.n_tr_tasks)
        self.val_task_indexes = np.arange(self.data_info.n_tr_tasks, self.data_info.n_tr_tasks + self.data_info.n_val_tasks)
        self.test_task_indexes = np.arange(self.data_info.n_tr_tasks + self.data_info.n_val_tasks, self.data_info.n_all_tasks)
        matrix_a = sp.linalg.sqrtm(matrix_w @ matrix_w.T)
        self.oracle = matrix_a / np.trace(matrix_a)

    def synthetic_data_gen_biased_sgd_paper(self):
        for task_idx in range(self.data_info.n_all_tasks):
            # generating and normalizing the inputs
            features = np.random.randn(self.data_info.n_all_points, self.data_info.n_dims)
            features = features / norm(features, axis=1, keepdims=True)

            # generating and normalizing the weight vectors
            weight_vector = np.random.normal(loc=4*np.ones(self.data_info.n_dims), scale=1).ravel()

            # generating labels and adding noise
            clean_labels = features @ weight_vector

            signal_to_noise_ratio = 10
            standard_noise = np.random.randn(self.data_info.n_all_points)
            noise_std = np.sqrt(np.var(clean_labels) / (signal_to_noise_ratio * np.var(standard_noise)))
            noisy_labels = clean_labels + noise_std * standard_noise

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

        self.tr_task_indexes = shuffled_task_indexes[:self.data_info.n_tr_tasks]
        self.val_task_indexes = shuffled_task_indexes[self.data_info.n_tr_tasks:self.data_info.n_tr_tasks + self.data_info.n_val_tasks]
        self.test_task_indexes = shuffled_task_indexes[self.data_info.n_tr_tasks + self.data_info.n_val_tasks:self.data_info.n_all_tasks]

    def movielens_gen(self):

        import scipy.io as sio
        temp = sio.loadmat('datasets/ml100kSparse.mat')
        full_matrix = temp['fullMatrix'].astype(float)

        # count the number each movie appears in the dataset and remove those that are too rare
        columns_to_keep = []
        for column in range(full_matrix.shape[1]):
            number_of_appearences = len(np.nonzero(full_matrix[:, column])[0])
            if number_of_appearences >= 20:
                columns_to_keep.append(column)
        full_matrix = full_matrix[:, columns_to_keep]

        n_movies = full_matrix.shape[1]
        self.data_info.n_dims = n_movies
        shuffled_task_indexes = np.random.permutation(self.data_info.n_all_tasks)

        for task_counter, user in enumerate(shuffled_task_indexes):
            # loading and normalizing the inputs
            zero_indexes = np.where(full_matrix[user, :].toarray() == 0)[1]
            non_zero_indexes = np.nonzero(full_matrix[user, :])[1]
            features = csc_matrix(np.eye(n_movies))
            features[zero_indexes, zero_indexes] = 0

            # loading the labels
            labels = full_matrix[user, :].toarray().ravel()

            if task_counter >= self.data_info.n_tr_tasks:
                # split into training and test
                tr_indexes, ts_indexes = train_test_split(non_zero_indexes, test_size=self.data_info.ts_points_pct)
                features_tr = deepcopy(features)
                labels_tr = deepcopy(labels)
                features_tr[ts_indexes, ts_indexes] = 0
                labels_tr[ts_indexes] = 0

                features_ts = deepcopy(features)
                labels_ts = deepcopy(labels)
                features_ts[tr_indexes, tr_indexes] = 0
                labels_ts[tr_indexes] = 0

                self.features_tr[user] = features_tr
                self.features_ts[user] = features_ts
                self.labels_tr[user] = labels_tr
                self.labels_ts[user] = labels_ts
            else:
                self.features_tr[user] = deepcopy(features)
                self.labels_tr[user] = deepcopy(labels)

        self.tr_task_indexes = shuffled_task_indexes[:self.data_info.n_tr_tasks]
        self.val_task_indexes = shuffled_task_indexes[self.data_info.n_tr_tasks:self.data_info.n_tr_tasks + self.data_info.n_val_tasks]
        self.test_task_indexes = shuffled_task_indexes[self.data_info.n_tr_tasks + self.data_info.n_val_tasks:self.data_info.n_all_tasks]
        self.full_matrix = full_matrix

    def jester_gen(self):

        import scipy.io as sio
        temp = sio.loadmat('datasets/' + self.data_info.dataset + 'Sparse.mat')
        full_matrix = temp[self.data_info.dataset + 'Sparse'].astype(float)

        n_jokes = full_matrix.shape[1]
        self.data_info.n_dims = n_jokes
        shuffled_task_indexes = np.random.permutation(self.data_info.n_all_tasks)

        for task_counter, user in enumerate(shuffled_task_indexes):
            # loading and normalizing the inputs
            zero_indexes = np.where(full_matrix[user, :].toarray() == 0)[1]
            non_zero_indexes = np.nonzero(full_matrix[user, :])[1]
            features = csc_matrix(np.eye(n_jokes))
            features[zero_indexes, zero_indexes] = 0

            # loading the labels
            labels = full_matrix[user, :].toarray().ravel()

            if task_counter >= self.data_info.n_tr_tasks:
                # split into training and test
                tr_indexes, ts_indexes = train_test_split(non_zero_indexes, test_size=self.data_info.ts_points_pct)
                features_tr = deepcopy(features)
                labels_tr = deepcopy(labels)
                features_tr[ts_indexes, ts_indexes] = 0
                labels_tr[ts_indexes] = 0

                features_ts = deepcopy(features)
                labels_ts = deepcopy(labels)
                features_ts[tr_indexes, tr_indexes] = 0
                labels_ts[tr_indexes] = 0

                self.features_tr[user] = features_tr
                self.features_ts[user] = features_ts
                self.labels_tr[user] = labels_tr
                self.labels_ts[user] = labels_ts
            else:
                self.features_tr[user] = deepcopy(features)
                self.labels_tr[user] = deepcopy(labels)

        self.tr_task_indexes = shuffled_task_indexes[:self.data_info.n_tr_tasks]
        self.val_task_indexes = shuffled_task_indexes[self.data_info.n_tr_tasks:self.data_info.n_tr_tasks + self.data_info.n_val_tasks]
        self.test_task_indexes = shuffled_task_indexes[self.data_info.n_tr_tasks + self.data_info.n_val_tasks:self.data_info.n_all_tasks]
        self.full_matrix = full_matrix

    def miniwikipedia_gen(self):
        data_path = 'datasets/miniwikinet/'

        def text2cbow(fname, w2v):
            """returns CBOW text representations from file with "label\ttoken token ... token\n" on each line
            Args:
                fname: file name
                w2v: {word: vector} dict
            Returns:
                numpy data array of shape [number of lines, vector dimension], numpy label array of shape [number of lines,]
            """
            try:
                f = open(fname, 'r', encoding='cp1252')
                loaded_labels, texts = zip(*(line.strip().split('\t') for line in f))
            except Exception as e:
                print('switching to utf8 | ' + str(e))
                f = open(fname, 'r', encoding='utf8')
                loaded_labels, texts = zip(*(line.strip().split('\t') for line in f))

            dims = len(w2v['god'])
            x_matrix = np.zeros((len(texts), dims))
            for text_idx, text in enumerate(texts):
                pool = []
                for w in text.split():
                    curr_embedding = w2v.get(w.lower())
                    if curr_embedding is not None:
                        pool.append(curr_embedding)
                        x_matrix[text_idx, :] = np.sum(pool, axis=0)

            nz = norm(x_matrix, axis=1) > 0.0
            x_matrix[nz] = normalize(x_matrix[nz])
            return x_matrix, np.array([int(label) for label in loaded_labels])

        def textfiles(corpus='bal', partition='train', m=32):
            """returns text file names
            Args:
                corpus: which subcorpus to use ('bal' or 'raw')
                partition: which partition to use ('train', 'dev', or 'test') ; ignored if corpus = 'raw'
                m: number of data points per class (1, 2, 4, ... , or 32) ; ignored if corpus = 'raw'
            Returns:
                list of filenames
            """

            datadir = data_path + corpus + '/'
            if corpus == 'bal':
                datadir += partition + '/' + str(m) + '/'
            return [datadir + fname for fname, _ in sorted(((fname, int(fname[:-4])) for fname in os.listdir(datadir)), key=itemgetter(1))]

        filenames_tr = textfiles(partition='train')
        filenames_val = textfiles(partition='dev')
        filenames_test = textfiles(partition='test')
        filenames = filenames_tr + filenames_val + filenames_test

        presaved_data = 'datasets/miniwikinet/presaved_50d_picklefile.dat'
        if os.path.isfile(presaved_data) is True:
            pack = pickle.load(open(presaved_data, "rb"))
            all_features, all_labels = pack
        else:
            w2v_dict = pickle.load(open('datasets/miniwikinet/glove.6B.50d_dict.pckl', "rb"))

            all_features = [None] * len(filenames)
            all_labels = [None] * len(filenames)
            for c_task in range(len(filenames)):
                x, y = text2cbow(filenames[c_task], w2v_dict)
                all_features[c_task] = x
                all_labels[c_task] = y
            pickle.dump([all_features, all_labels], open(presaved_data, "wb"))

        self.data_info.n_dims = all_features[0].shape[1]
        shuffled_task_indexes = np.random.permutation(self.data_info.n_all_tasks)

        for task_counter, task in enumerate(shuffled_task_indexes):
            # loading and normalizing the inputs
            features = all_features[task]
            features = features

            # loading the labels
            labels = all_labels[task].ravel()

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

        self.tr_task_indexes = shuffled_task_indexes[:self.data_info.n_tr_tasks]
        self.val_task_indexes = shuffled_task_indexes[self.data_info.n_tr_tasks:self.data_info.n_tr_tasks + self.data_info.n_val_tasks]
        self.test_task_indexes = shuffled_task_indexes[self.data_info.n_tr_tasks + self.data_info.n_val_tasks:self.data_info.n_all_tasks]
