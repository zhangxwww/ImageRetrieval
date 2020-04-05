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


def save_match_result(query_name, matches):
    pass


def save_precision_result(precision):
    pass


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
