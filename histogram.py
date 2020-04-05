import numpy as np

_PARTITION_16 = np.array([128, 64, 128], dtype=np.uint8).reshape((1, 1, -1))
_PARTITION_128 = np.array([64, 32, 64], dtype=np.uint8).reshape((1, 1, -1))

_WEIGHT_16 = np.array([8, 2, 1], dtype=np.uint8)
_WEIGHT_128 = np.array([32, 4, 1], dtype=np.uint8)


def histogram(img, partition):
    p = _PARTITION_16 if partition == 16 else _PARTITION_128
    w = _WEIGHT_16 if partition == 16 else _WEIGHT_128
    interval = img // p
    interval = np.einsum('ijk,k->ij', interval, w)
    res = [np.sum(interval == i) for i in range(partition)]

    return np.array(res)
