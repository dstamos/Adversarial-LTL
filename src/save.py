import os
import pickle
import logging
import time


class Logger:
    def __init__(self, data_info, training_info):
        self.data_info = data_info
        self.training_info = training_info
        self.results_filename = "seed_" + str(data_info.seed)
        self.results_foldername = 'results/' + data_info.dataset + \
                                  '-T_' + str(data_info.n_tr_tasks) + \
                                  '-v_' + str(data_info.val_points_pct) + \
                                  '/' + training_info.method

        if not os.path.exists('results'):
            os.makedirs('results')

        if not os.path.exists('logs'):
            os.makedirs('logs')
        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            filename="logs/%(date_time)s_" + self.results_filename + ".log" % {'date_time': time.strftime("%Y-%m-%d_%H.%M.%S")}, level=logging.INFO)

    def save(self, results):
        if not os.path.exists(self.results_foldername):
            os.makedirs(self.results_foldername)
        f = open(self.results_foldername + '/' + self.results_foldername + ".pckl", 'wb')
        pickle.dump(results, f)
        pickle.dump(self.data_info, f)
        pickle.dump(self.training_info, f)
        f.close()

    @staticmethod
    def log_specific_object(logging_object):
        # Logging_object is a string
        logging.critical(logging_object)
