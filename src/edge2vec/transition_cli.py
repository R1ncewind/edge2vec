# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np

from edge2vec.transition import calculate_edge_transition_matrix
from edge2vec.utils import read_graph

HERE = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, 'data'))


def parse_args():
    """Parses the node2vec arguments."""
    parser = argparse.ArgumentParser(description="Run edge transition matrix.")

    parser.add_argument('--input', nargs='?', default=os.path.join(DATA_DIR, 'data.csv'),
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default=os.path.join(DATA_DIR, 'matrix.txt'),
                        help='store transition matrix')

    parser.add_argument('--em_iteration', default=10, type=int,
                        help='EM iterations for transition matrix. Default: %(default)')

    parser.add_argument('--e_step', default=3, type=int,
                        help='E step in the EM algorithm: there are four expectation metrics')

    parser.add_argument('--dimensions', type=int, default=10,
                        help='Number of dimensions. Default: %(default)')

    parser.add_argument('--walk-length', type=int, default=3,
                        help='Length of walk per source. Default: %(default)')

    parser.add_argument('--walk-sample-size', type=int, default=1000,
                        help='Number of walks to sample on each epoch.')

    parser.add_argument('--number-walks', type=int, default=50,
                        help='Number of walks per source. Default: %(default)')

    parser.add_argument('--window-size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')

    parser.add_argument('--iter', default=10, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1.0,
                        help='Return hyper-parameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1.0,
                        help='In/out hyper-parameter. Default is 1.')

    # dest='weighted' means the arg parameter name is weighted.
    # There is only one parameter: args.weighted
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='weighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='directed', action='store_false')
    parser.set_defaults(directed=False)

    parser.add_argument('--mp', dest='use_multiprocessing', action='store_true',
                        help='Use multiprocessing.')

    return parser.parse_args()


def main():
    args = parse_args()
    graph = read_graph(
        path=args.input,
        weighted=args.weighted,
        directed=args.directed,
    )
    if args.walk_sample_size > graph.number_of_edges():
        print('Can not sample more edges than graph has.')
        args.walk_sample_size = graph.number_of_edges()

    trans_matrix = calculate_edge_transition_matrix(
        graph=graph,
        e_step=args.e_step,
        em_iteration=args.em_iteration,
        directed=args.directed,
        walk_sample_size=args.walk_sample_size,
        p=args.p,
        q=args.q,
        walk_epochs=args.number_walks,
        walk_length=args.walk_length,
        use_multiprocessing=args.use_multiprocessing,
    )
    np.savetxt(args.output, trans_matrix)


if __name__ == "__main__":
    main()
