class DataSettings:
    def __init__(self, data_dict):
        self.seed = data_dict['seed']
        self.dataset = data_dict['dataset']
        self.n_all_tasks = data_dict['n_all_tasks']
        self.n_tr_tasks = data_dict['n_tr_tasks']
        self.n_val_tasks = data_dict['n_val_tasks']
        self.n_test_tasks = data_dict['n_all_tasks'] - (data_dict['n_tr_tasks'] + data_dict['n_val_tasks'])

        self.n_all_points = data_dict['n_all_points']
        self.ts_points_pct = data_dict['ts_points_pct']

        if self.dataset == 'synthetic':
            self.n_dims = data_dict['n_dims']
            self.noise_std = data_dict['noise_std']


class TrainingSettings:
    def __init__(self, training_dict):
        self.method = training_dict['method']
        self.meta_algo_regul_param = training_dict['meta_algo_regul_param']
        self.convergence_tol = training_dict['convergence_tol']

        if self.method == 'temp_method':
            self.inner_regul_param = training_dict['inner_regul_param']
