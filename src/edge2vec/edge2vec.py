'''
use existing matrix to run edge2vec
'''

import argparse
import random
from typing import Optional

import numpy as np
from gensim.models import Word2Vec

from .utils import read_graph


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run edge transition matrix.")

    parser.add_argument('--input', nargs='?', default='weighted_graph.txt',
                        help='Input graph path')

    parser.add_argument('--matrix', nargs='?', default='matrix.txt',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='vector.txt',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=3,
                        help='Length of walk per source. Default is 3.')

    parser.add_argument('--num-walks', type=int, default=2,
                        help='Number of walks per source. Default is 2.')

    parser.add_argument('--window-size', type=int, default=2,
                        help='Context size for optimization. Default is 2.')

    parser.add_argument('--iter', default=5, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='In/out hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='weighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='directed', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def get_walks(graph, num_walks, walk_length, matrix, p, q):
    """Generate random walk paths constrainted by transition matrix."""
    walks = []
    nodes = list(graph.nodes())
    print('Walk iteration:')
    for walk_iter in range(num_walks):
        print(str(walk_iter + 1), '/', str(num_walks))
        random.shuffle(nodes)
        for node in nodes:
            # print "chosen node id: ",nodes
            walks.append(_get_walk(graph, walk_length, node, matrix, p, q))
    return walks


def _get_walk(graph, walk_length, start_node, matrix, p, q):
    """Return a random walk path."""
    walk = [start_node]
    while len(walk) < walk_length:  # here we may need to consider some dead end issues
        cur = walk[-1]
        cur_nbrs = sorted(graph.neighbors(cur))  # (G.neighbors(cur))

        if len(cur_nbrs) == 0:
            return walk  # the walk has hit a dead end
        random.shuffle(cur_nbrs)
        if len(walk) == 1:
            rand = int(np.random.rand() * len(cur_nbrs))
            next = cur_nbrs[rand]
            walk.append(next)
        else:
            prev = walk[-2]

            pre_edge_type = graph.edges[prev, cur]['type'] - 1

            distance_sum = 0
            for neighbor in cur_nbrs:
                neighbor_link = graph.edges[cur, neighbor]
                # print "neighbor_link: ",neighbor_link
                neighbor_link_type = neighbor_link['type']
                # print "neighbor_link_type: ",neighbor_link_type
                neighbor_link_weight = neighbor_link['weight']
                transition_probability = matrix[pre_edge_type][neighbor_link_type - 1]

                if graph.has_edge(neighbor, prev) or graph.has_edge(prev, neighbor):  # undirected graph
                    distance_sum += transition_probability * neighbor_link_weight / p  # +1 normalization
                elif neighbor == prev:  # decide whether it can random walk back
                    distance_sum += transition_probability * neighbor_link_weight
                else:
                    distance_sum += transition_probability * neighbor_link_weight / q

            '''
            pick up the next step link
            '''

            rand = np.random.rand() * distance_sum
            threshold = 0
            for neighbor in cur_nbrs:
                neighbor_link = graph[cur][neighbor]
                # print "neighbor_link: ",neighbor_link
                neighbor_link_type = neighbor_link['type']
                # print "neighbor_link_type: ",neighbor_link_type
                neighbor_link_weight = neighbor_link['weight']
                transition_probability = matrix[pre_edge_type - 1][neighbor_link_type - 1]

                if graph.has_edge(neighbor, prev) or graph.has_edge(prev, neighbor):  # undirected graph

                    threshold += transition_probability * neighbor_link_weight / p
                    if threshold >= rand:
                        next = neighbor
                        break
                elif neighbor == prev:
                    threshold += transition_probability * neighbor_link_weight
                    if threshold >= rand:
                        next = neighbor
                        break
                else:
                    threshold += transition_probability * neighbor_link_weight / q
                    if threshold >= rand:
                        next = neighbor
                        break

            walk.append(next)

        # print "walk length: ",len(walk),walk
        # print "edge walk: ",len(edge_walk),edge_walk 
    return walk


def main_helper(args):
    transition_matrix = np.loadtxt(args.matrix, delimiter=' ')
    graph = read_graph(
        path=args.input,
        weighted=args.weighted,
        directed=args.directed,
    )
    model = train(
        transition_matrix=transition_matrix,
        graph=graph,
        number_walks=args.num_walks,
        walk_length=args.walk_length,
        p=args.p,
        q=args.q,
        window=args.window_size,
        size=args.dimensions,
    )
    model.wv.save_word2vec_format(args.output)


def train(
        *,
        transition_matrix,
        graph,
        number_walks: Optional[int] = None,
        walk_length: Optional[int] = None,
        p: Optional[float] = None,
        q: Optional[float] = None,
        **word2vec_kwargs
) -> Word2Vec:
    walks = get_walks(
        graph,
        number_walks,
        walk_length,
        transition_matrix,
        p,
        q,
    )
    return Word2Vec(list(walks), **word2vec_kwargs)


def main():
    args = parse_args()
    main_helper(args)


if __name__ == "__main__":
    main()
