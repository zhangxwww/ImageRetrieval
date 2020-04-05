import numpy as np


def L2(p, q):
    """
    :param p: n * d
    :param q: m * d
    :return:  n * m
    """
    _, _, n, m, pp, qq = _prepare(p, q)
    delta = pp - qq
    return np.sqrt(np.sum(delta * delta, axis=2).reshape((n, m)))


def HI(p, q):
    """
    :param p: n * d
    :param q: m * d
    :return:  n * m
    """
    _, q, n, m, pp, qq = _prepare(p, q)
    pp = pp.repeat(m, 1)  # n * m * d
    qq = qq.repeat(n, 0)  # n * m * d
    rr = np.stack([pp, qq], axis=0)  # 2 * n * m * d
    min_ = rr.min(axis=0)  # n * m * d
    dividend = min_.sum(axis=2)  # n * m
    divider = q.sum(axis=1).reshape((1, -1))  # 1 * m
    return dividend / divider


def BH(p, q):
    """
    :param p: n * d
    :param q: m * d
    :return:  n * m
    """
    _, _, _, _, pp, qq = _prepare(p, q)
    sqrt = np.sqrt(pp * qq)  # n * m * d
    return np.sqrt(1 - np.sum(sqrt, axis=2))


def _prepare(p, q):
    p = p.astype(np.float)
    q = q.astype(np.float)
    n = p.shape[0]
    m = q.shape[0]
    pp = p.reshape((n, 1, -1))
    qq = q.reshape((1, m, -1))
    return p, q, n, m, pp, qq


METRICS = {
    'L2': L2,
    'HI': HI,
    'BH': BH
}
