import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")


def main(method_id):

    if DATASET_IDX == 0:
        dataset = 'synthetic'
    elif DATASET_IDX == 1:
        dataset = 'schools'
    elif DATASET_IDX == 2:
        dataset = 'movielens100k'
    else:
        raise ValueError

    the_table = [None] * len(SEED_RANGE)

    if method_id == 0:
        method = 'ITL_SGD'
    elif method_id == 1:
        method = 'ITL_ERM'
    elif method_id == 2:
        method = 'LTL_SGD-SGD'
    elif method_id == 3:
        method = 'LTL_ERM-SGD'
    elif method_id == 4:
        method = 'LTL_Oracle-SGD'
    else:
        raise ValueError

    for seed_idx, seed in enumerate(SEED_RANGE):
        best_score = np.Inf
        best_scores = None
        for inner_param_idx, inner_param in enumerate(INNER_PARAM_RANGE):
            for meta_param_idx, meta_param in enumerate(META_PARAM_RANGE):
                results_filename = "seed_" + str(seed) + '_' + str(inner_param) + '_' + str(meta_param)
                results_full_path = 'results/' + dataset + '/' + method

                try:
                    results = extract_results(results_full_path, results_filename)
                except Exception as e:
                    print(inner_param_idx, inner_param_idx, seed)
                    continue

                test_score = np.mean(results['test_scores'])

                if test_score < best_score:
                    best_score = test_score
                    best_scores = results['test_scores']

                if (method_id == 0) or (method_id == 1):
                    the_table[seed_idx] = [best_scores] * 3000
                else:
                    the_table[seed_idx] = best_scores

        bucket = [the_table[seed_idx][0]]
        for idx in range(len(the_table[seed_idx])):
            if idx > 0:
                bucket.append(np.nanmean(the_table[seed_idx][:idx]))
        the_table[seed_idx] = bucket

    the_table = [x for x in the_table if x is not None]
    average_shit = np.nanmean(the_table, axis=0)
    errors = np.nanstd(the_table, axis=0)

    # try:
    #     average_shit = np.nanmean(the_table, axis=0)
    #     errors = np.nanstd(the_table, axis=0)
    #     good_indexes = 'spam'
    # except:
    #     # hardcoded fix for the plot of vectors with none values
    #     ruler = np.array(the_table[0])
    #     good_indexes = np.where(ruler != None)[0]
    #     bad_indexes = np.where(ruler == None)[0]
    #     the_table = np.array(the_table).astype(float)
    #     average_shit = np.nanmean(the_table[:, good_indexes], axis=0)
    #     errors = np.nanstd(the_table[:, good_indexes], axis=0)

    return average_shit, errors


def extract_results(results_full_path, results_filename):
    f = open('C:/cluster/LTL/adversarial/' + results_full_path + '/' + results_filename + ".pckl", 'rb')
    results = pickle.load(f)
    f.close()
    return results


if __name__ == "__main__":

    # SEED_RANGE = [1, 2, 3, 4, 5]
    SEED_RANGE = [1, 2]
    DATASET_IDX = 0
    METHOD_RANGE = [0, 1, 2, 3]

    colors = ['#3b79cc', '#d84141', '#e58f39', '#489e3f']
    linestyles = ['-', '--', '-.', '-.']
    legend_array = ['ITL SGD', 'ITL ERM', 'LTL SGD-SGD', 'LTL ERM-SGD']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), facecolor='white')
    fontsize = 50

    for method_idx, method_num in enumerate(METHOD_RANGE):
        INNER_PARAM_RANGE = [10 ** float(i) for i in np.linspace(-6, 3, 20)]
        if (method_num == 0) or (method_num == 1):
            META_PARAM_RANGE = [np.nan]
        else:
            META_PARAM_RANGE = [10 ** float(i) for i in np.linspace(-4, 3, 10)]

        c = colors[method_num]
        line_style = linestyles[method_num]

        average_perf, std_perf = main(method_num)

        ax.plot(range(len(average_perf)), average_perf, linewidth=4, color=c, linestyle=line_style)
        ax.fill_between(range(len(average_perf)), average_perf - std_perf, average_perf + std_perf, alpha=0.1, edgecolor=c, facecolor=c, antialiased=True, label='_nolegend_')

        ax.set_xlabel('# training tasks', fontsize=fontsize, fontweight="bold")
        ax.set_ylabel('mean test MAE', fontsize=fontsize, fontweight="bold")

        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize)

    plt.legend([legend_array[i] for i in METHOD_RANGE], fontsize=fontsize)
    plt.tight_layout()

    plt.savefig('plot_' + str(DATASET_IDX) + '-fontsize_' + str(fontsize) + '.png', format='png', dpi=200)
    plt.pause(0.001)
    print('done')
