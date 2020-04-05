import numpy as np
from dataset import Dataset
from histogram import histogram, PARTITIONS
from metric import METRICS
from utils import topk, save_match_result, save_precision_result, Precision

K = 30


def main():
    dataset = Dataset()
    all_images, all_image_names = dataset.get_all_images()
    query_images, query_image_names = dataset.get_query_images()
    for p in PARTITIONS:
        all_feature = np.stack([histogram(i, p) for i in all_images], axis=0)
        query_feature = np.stack([histogram(i, p) for i in query_images], axis=0)
        for m in METRICS:
            run_exp(p, m, {
                'all_feature': all_feature,
                'query_feature': query_feature,
                'all_name': all_image_names,
                'query_name': query_image_names
            })


def run_exp(partition, metric, data):
    af = data['all_feature']
    qf = data['query_feature']
    an = data['all_name']
    qn = data['query_name']
    distance = metric(af, qf)
    closest = topk(distance, K, axis=0)
    n_query = qf.shape[0]
    precision = []
    for i in range(n_query):
        index = closest[:, i]
        matches = []
        p = Precision(qn[i])
        for idx in index:
            matches.append((an[idx], distance[idx, i]))
            p.check(an[idx])
        save_match_result(qn[i], matches)
        precision.append((qn[i], p.get_precision()))
    save_precision_result(precision)


if __name__ == '__main__':
    main()
