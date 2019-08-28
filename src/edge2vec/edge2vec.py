'''
use existing matrix to run edge2vec
'''

import argparse
import random
from typing import Optional, Callable, Iterable, List
import logging

import numpy as np
from gensim.models import Word2Vec

from .utils import read_graph
from tqdm import trange, tqdm

from multiprocessing import Pool, cpu_count
from functools import partial

logger = logging.getLogger(__name__)
Walk = List[int]


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


def get_walks(graph, num_walks, walk_length, matrix, p, q, use_multiprocessing: bool = True, ):
    """Generate random walk paths constrained by transition matrix"""
    
    nodes = list(graph.nodes())

    shuffled_nodes = random.shuffle(nodes)
    iterated_nodes = iterator_node(shuffled_nodes, num_walks)
    for node in iterated_nodes:
        print(node)
    partial_get_walk = partial(_get_walk, graph, walk_length, matrix, p, q)
    if use_multiprocessing:
        with Pool(cpu_count()) as p:
            logger.warning(f'Use multiprocessing on {cpu_count()} cores')
            walks = p.map(partial_get_walk, iterated_nodes)
    else:
        walks = map(partial_get_walk, iterated_nodes)

    return walks


def iterator_node(nodes, num_walks):
    for iter_walk in range(0, num_walks):
        for node in nodes:
            yield node


def _get_walk(start_node, graph, walk_length, matrix, p, q):
    """Return a random walk path."""
    walk = [start_node]
    prev = None
    while len(walk) < walk_length:  # here we may need to consider some dead end issues
        cur = walk[-1]
        cur_nbrs = list(graph.neighbors(cur))  # (G.neighbors(cur))

        if len(cur_nbrs) == 0:
            return walk  # the walk has hit a dead end
        random.shuffle(cur_nbrs)
        if len(walk) == 1:
            walk.append(random.choice(cur_nbrs))
        else:
            prev = walk[-2]

            if prev not in graph:
                print(f'Problem: prev not in graph: {prev}')
                raise ValueError
            elif cur not in graph[prev]:
                print(f'Problem: cur not in graph: {cur}')
                print(list(graph[prev].keys()))
                raise ValueError

            pre_edge_type = graph[prev][cur]['type'] - 1

            distance_sum = 0

            for neighbor in cur_nbrs:
                # print "neighbor_link: ",neighbor_link
                neighbor_link_type = graph[cur][neighbor]['type'] - 1
                # Get transition probability based on the previous edge and the current possible edge
                transition_probability = matrix[pre_edge_type][neighbor_link_type]

                neighbor_link_weight = graph[cur][neighbor]['weight']

                if graph.has_edge(neighbor, prev) or graph.has_edge(prev, neighbor):  # undirected graph
                    distance_sum += transition_probability * neighbor_link_weight / p  # +1 normalization
                elif neighbor == prev:  # decide whether it can random walk back
                    distance_sum += transition_probability * neighbor_link_weight
                else:  # Triangle
                    distance_sum += transition_probability * neighbor_link_weight / q

            '''
            pick up the next step link
            '''
            nn = pick_neighbors(graph, cur, prev, cur_nbrs, pre_edge_type, matrix, distance_sum, p, q)
            if nn is not None:
                walk.append(nn)
            else:
                print('No neighbour to go!')
                print(prev, cur)
                walk.append(random.choice(cur_nbrs))

        # print "walk length: ",len(walk),walk
        # print "edge walk: ",len(edge_walk),edge_walk 
    return walk


def pick_neighbors(graph, cur, prev, neighbors, pre_edge_type, matrix, d, p, q):
    rand = np.random.rand() * d
    threshold = 0
    for neighbor in neighbors:
        neighbor_link = graph[cur][neighbor]
        # print "neighbor_link: ",neighbor_link
        neighbor_link_type = neighbor_link['type'] - 1
        # print "neighbor_link_type: ",neighbor_link_type
        neighbor_link_weight = neighbor_link['weight'] - 1
        transition_probability = matrix[pre_edge_type][neighbor_link_type]

        if graph.has_edge(neighbor, prev) or graph.has_edge(prev, neighbor):  # undirected graph
            threshold += transition_probability * neighbor_link_weight / p
        elif neighbor == prev:
            threshold += transition_probability * neighbor_link_weight
        else:
            threshold += transition_probability * neighbor_link_weight / q

        if threshold >= rand:
            return neighbor


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
        window: Optional[int] = None,
        size: Optional[int] = None,
) -> Word2Vec:
    walks = get_walks(
        graph,
        number_walks,
        walk_length,
        transition_matrix,
        p,
        q,
    )
    return Word2Vec(
        list(walks),
        size=size or 100,
        window=window or 5,
    )


def main():
    args = parse_args()
    main_helper(args)


if __name__ == "__main__":
    main()
