class DataSettings:
    def __init__(self, data_dict):
        self.seed = data_dict['seed']
        self.dataset = data_dict['dataset']
        self.n_tr_tasks = data_dict['n_tr_tasks']
        self.n_val_tasks = data_dict['n_val_tasks']
        self.n_test_tasks = data_dict['n_test_tasks']
        self.n_all_tasks = data_dict['n_tr_tasks'] + data_dict['n_val_tasks'] + data_dict['n_test_tasks']

        self.ts_points_pct = data_dict['ts_points_pct']

        if self.dataset == 'synthetic' or self.dataset == 'synthetic_data_gen_biased_sgd_paper':
            self.n_all_points = data_dict['n_all_points']
            self.n_dims = data_dict['n_dims']
            self.noise_std = data_dict['noise_std']


class TrainingSettings:
    def __init__(self, training_dict):
        self.method = training_dict['method']
        self.inner_regul_param = training_dict['inner_regul_param']
        self.meta_algo_regul_param = training_dict['meta_algo_regul_param']

        if self.method == 'LTL_SGD-SGD' or self.method == 'LTL_ERM-SGD' or self.method == 'LTL_Oracle-SGD' or self.method == 'LTL_ERM-ERM':
            self.meta_algo_regul_param = training_dict['meta_algo_regul_param']
