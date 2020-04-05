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
        for m in METRICS.items():
            run_exp({
                'partition': p,
                'metric': m
            }, {
                'all_feature': all_feature,
                'query_feature': query_feature,
                'all_name': all_image_names,
                'query_name': query_image_names
            })


def run_exp(exp_info, data):
    af = data['all_feature']
    qf = data['query_feature']
    an = data['all_name']
    qn = data['query_name']
    metric_name, metric = exp_info['metric']
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
        save_match_result(qn[i], matches, exp_info)
        precision.append((qn[i], p.get_precision()))
    avg_precision = sum([p[1] for p in precision]) / len(precision)
    save_precision_result(precision, avg_precision, exp_info)


if __name__ == '__main__':
    main()
