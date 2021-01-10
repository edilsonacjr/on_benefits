"""
    Brief description
    
    Author: Edilson A. Correa Junior <edilsonacjr@gmail.com>
"""
import sys
import argparse

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn import metrics
from sklearn.metrics import pairwise_distances

from igraph import Graph, ADJ_UNDIRECTED

from utils import to_xnet


def detect_community(g, method):
    if method == 'community_multilevel' or method == 'community_leading_eigenvector':
        comm = getattr(g, method)()
    else:
        comm = getattr(g, method)()
        comm = comm.as_clustering()
    y_pred = comm.membership

    return y_pred, comm.modularity


def gen_nets_comm(ks,
                  data_path=None,
                  data_file_name='data.npy',
                  target_file_name='target.npy',
                  metric='euclidean',
                  community_method='community_multilevel',
                  n_jobs=-1):
    if not data_path:
        raise ValueError('data_path not specified.')
    path = Path(data_path)

    x = np.load(path / data_file_name)
    y = np.load(path / target_file_name)

    similarity_matrix = 1 / (1 + pairwise_distances(x, metric=metric, n_jobs=n_jobs))
    result_modularity = []
    result_ari = []
    result_nmi = []
    connec_point = None
    for k in ks:
        M = similarity_matrix.copy()
        to_remove = M.shape[0] - (k + 1)  # add 1 to eliminate loops
        for vec in M:
            vec[vec.argsort()[:to_remove]] = 0

        g = Graph.Weighted_Adjacency(M.tolist(), mode=ADJ_UNDIRECTED, loops=False)
        g.vs['name'] = y

        # Verify in which k the network is connected
        if not connec_point and not g.is_connected():
            connec_point = k

        y_pred, modularity = detect_community(g, community_method)

        path_save = Path(path) / 'nets'
        path_save.mkdir(parents=True, exist_ok=True)
        net_name = 'net_%s_k_%i.xnet' % (metric, k)
        labels_name = 'net_%s_k_%i_labels_comm.txt' % (metric, k)

        to_xnet(g, path_save / net_name, names=True)
        np.savetxt(path_save / labels_name, y_pred, fmt='%s')

        metrics.adjusted_rand_score(y, y_pred)

        result_modularity.append(modularity)
        result_ari.append(metrics.adjusted_rand_score(y, y_pred))
        result_nmi.append(metrics.normalized_mutual_info_score(y, y_pred))

    path_results = path / 'results'
    path_results.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({'NMI': result_nmi, 'ARI': result_ari, 'Modularity': result_modularity})
    df.to_csv(path_results / ('%s.csv' % metric))
    df.index = sorted(ks) #df.index + 1
    plot = df.plot(xticks=[1] + list(range(0, max(ks) + 1, 5))[1:], ylim=(0, 1), use_index=True)
    plot.set_xlabel('k')

    plot.axvline(connec_point, color='k', linestyle='--')
    plot.text(connec_point + 0.01, 0.98, 'connected', rotation=90)

    fig = plot.get_figure()
    fig.savefig(path_results / ('%s.pdf' % metric))


def main(args):
    np.random.seed(1)

    gen_nets_comm(args.ks,
                  data_path=args.data_path,
                  data_file_name=args.data_file_name,
                  target_file_name=args.target_file_name,
                  metric=args.metric,
                  community_method=args.comm_method,
                  n_jobs=args.n_jobs)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, help='Directory where the dataset is stored.',
                        default='data/olivetti')
    parser.add_argument('--data_file_name', type=str, help='Data file name.', default='data.npy')
    parser.add_argument('--target_file_name', type=str, help='Data file name.', default='target.npy')
    parser.add_argument('--metric', type=str, help='Valid distances for sklearn.metrics.pairwise_distances',
                        default='euclidean')
    parser.add_argument('--n_jobs', type=int, help='Number of cores to be used in metric computation, -1 uses all ',
                        default=-1)
    parser.add_argument('--comm_method', type=str, choices=['community_multilevel', 'community_leading_eigenvector',
                                                            'community_fastgreedy', 'community_walktrap'],
                        help='Community method to use.', default='community_multilevel')
    parser.add_argument('--ks', nargs='+', type=int, help='<Required> Set flag', default=[5, 15, 30, 50, 100])

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
