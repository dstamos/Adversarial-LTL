import os
import pickle
import logging
import time


class Logger:
    def __init__(self, data_info, training_info, inner_param, outer_param):
        self.data_info = data_info
        self.training_info = training_info
        self.results_filename = "seed_" + str(data_info.seed) + '_' + str(inner_param) + '_' + str(outer_param)
        self.results_full_path = 'results/' + data_info.dataset + '/' + training_info.method

        if not os.path.exists('results'):
            os.makedirs('results')

        if not os.path.exists('logs'):
            os.makedirs('logs')
        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            filename="logs/%(date_time)s_" + self.results_filename + ".log" % {'date_time': time.strftime("%Y-%m-%d_%H.%M.%S")}, level=logging.INFO)

    def save(self, results):
        if not os.path.exists(self.results_full_path):
            os.makedirs(self.results_full_path)
        f = open(self.results_full_path + '/' + self.results_filename + ".pckl", 'wb')
        pickle.dump(results, f)
        pickle.dump(self.data_info, f)
        pickle.dump(self.training_info, f)
        f.close()

    @staticmethod
    def log_event(logging_object):
        # Logging_object is a string
        print(logging_object)
        logging.critical(logging_object)
