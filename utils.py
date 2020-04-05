import os
import numpy as np


def topk(matrix, k, axis=0):
    a_part = np.argpartition(matrix, k, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:k, :], row_index], axis=axis)
        return a_part[0:k, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:k]], axis=axis)
        return a_part[:, 0:k][column_index, a_sec_argsort_K]


def save_match_result(query_name, matches, exp_info):
    save_dir = _get_save_dir(exp_info)
    result = ['{} {}\n'.format(name, dist) for name, dist in matches]
    filename = _get_query_result_filename(query_name)
    filename = os.path.join(save_dir, filename)
    with open(filename, 'w') as f:
        f.writelines(result)


def save_precision_result(precision, avg_precision, exp_info):
    save_dir = _get_save_dir(exp_info)
    result = ['{} {}\n'.format(name, p) for name, p in precision]
    filename = 'res_overall.txt'
    filename = os.path.join(save_dir, filename)
    with open(filename, 'w') as f:
        f.writelines(result)
        f.write('{}\n'.format(avg_precision))


class Precision:
    def __init__(self, qname):
        self.category = qname.split('/')[0]
        self.correct = 0
        self.total = 0

    def check(self, aname):
        c = aname.split('/')[0]
        if c == self.category:
            self.correct += 1
        self.total += 1

    def get_precision(self):
        return self.correct / self.total


def _get_save_dir(exp_info):
    dir_name = '{}_{}'.format(exp_info['partition'], exp_info['metric'][0])
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def _get_query_result_filename(qn):
    category, name = qn.split('/')
    name = name.split('.')[0]
    return 'res_{}_{}.txt'.format(category, name)
