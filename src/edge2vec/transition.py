# -*- coding: utf-8 -*-

"""
first version: unweighted, undirected network
use edge random walk to generate edge transition matrix based on EM algorithm
"""

import itertools
import logging
from collections import Counter, defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Any, Iterable, List, Mapping, Optional, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm, trange

from .math import entroy_test, pearsonr_test, spearmanr_test, wilcoxon_test

Edge = Tuple[int, int, Mapping[str, Any]]
Walk = List[int]

logger = logging.getLogger(__name__)


def calculate_edge_transition_matrix(
    *,
    graph,
    directed: bool,
    e_step: int,
    em_iteration: int,
    walk_epochs: Optional[int] = None,
    walk_length: Optional[int] = None,
    p: Optional[float] = None,
    q: Optional[float] = None,
    walk_sample_size: Optional[int] = None,
    use_multiprocessing: bool = True,
) -> np.ndarray:
    # print "------begin to write graph---------"
    # generate_graph_write_edgelist(args.m1,args.m2,args.input)

    edge_types = {
        data['type']
        for _, _, data in graph.edges(data=True)
    }
    number_edge_types = len(edge_types)

    matrix = np.ones(shape=(number_edge_types, number_edge_types)) / (number_edge_types * number_edge_types)

    if e_step == 1 or not e_step:  # default
        evaluation_test = wilcoxon_test
    elif e_step == 2:
        evaluation_test = entroy_test
    elif e_step == 3:
        evaluation_test = spearmanr_test
    elif e_step == 4:
        evaluation_test = pearsonr_test
    else:
        raise ValueError('not correct evaluation metric! You need to choose from 1-4')

    it = trange(em_iteration, desc='Expectation/Maximization')
    for _ in it:
        # M step
        walks: Iterable[Walk] = get_edge_walks(
            graph=graph,
            walk_epochs=walk_epochs,
            walk_length=walk_length,
            matrix=matrix,
            p=p,
            q=q,
            is_directed=directed,
            walk_sample_size=walk_sample_size,
            use_multiprocessing=use_multiprocessing,
        )

        # E step
        matrix = update_trans_matrix(
            walks=walks,
            number_edge_types=number_edge_types,
            evaluation_test=evaluation_test,
        )

    return matrix


def get_edge_walks(
    *,
    graph,
    walk_epochs: int,
    walk_length: int,
    matrix,
    is_directed,
    p,
    q,
    walk_sample_size: Optional[int] = None,
    use_multiprocessing: bool = True,
) -> Iterable[Walk]:
    """Generate random walk paths constrained by transition matrix."""
    if walk_sample_size is None:
        walk_sample_size = 1000

    if walk_sample_size > graph.number_of_edges():
        logger.warning('Can not sample more edges than graph has.')
        links: Iterable[Edge] = graph.edges(data=True)
    else:
        links: Iterable[Edge] = _iterate_links(graph, walk_epochs, walk_sample_size)

    _partial_get_edge_walk = partial(
        _get_edge_walk,
        graph=graph,
        walk_length=walk_length,
        matrix=matrix,
        is_directed=is_directed,
        p=p,
        q=q,
    )

    if not use_multiprocessing:
        return map(_partial_get_edge_walk, links)

    with Pool(cpu_count()) as pool:
        logger.info(f'Use multiprocessing on {cpu_count()} cores')
        return pool.map(_partial_get_edge_walk, links)


def _iterate_links(graph: nx.Graph, n_iter: int, n_links: int) -> Iterable[Edge]:
    links: List[Edge] = list(graph.edges(data=True))
    for _ in range(n_iter):
        for i in np.random.choice(len(links), size=n_links, replace=False):
            yield links[i]


def _get_edge_walk(
    start_link: Edge,
    *,
    graph,
    walk_length,
    matrix,
    is_directed,
    p,
    q,
) -> Walk:
    """Return a random walk path of types"""
    # print "start link: ", type(start_link), start_link
    walk = [start_link]
    result = [get_type_from_link(start_link)]
    # print "result ",result
    while len(walk) < walk_length:  # here we may need to consider some dead end issues
        cur = walk[-1]
        start_node = cur[0]
        end_node = cur[1]
        cur_edge_type = cur[2]['type']

        '''
        find the direction of link to go. If a node degree is 1, it means if go that direction, there is no other 
        links to go further if the link are the only link for both nodes, the link will have no neighbours (need to
        have teleportation later)
        '''
        '''
        consider the hub nodes and reduce the hub influence
        '''
        if is_directed:  # directed graph has random walk direction already
            direction_node = end_node
            left_node = start_node
        else:  # for undirected graph, first consider the random walk direction by choosing the start node
            start_direction = 1.0 / graph.degree(start_node)
            end_direction = 1.0 / graph.degree(end_node)
            prob = start_direction / (start_direction + end_direction)
            # print "start node: ", start_node, " degree: ", G.degree(start_node)
            # print "end node: ", end_node, " degree: ", G.degree(end_node)

            # print cur[0], cur[1]
            rand = np.random.rand()
            # print "random number ",rand
            # print "probability for start node: ",prob

            if prob >= rand:
                # print "yes"
                direction_node = start_node
                left_node = end_node
            else:
                direction_node = end_node
                left_node = start_node
        # print "directed node: ",direction_node
        # print "left_node node: ",left_node
        '''
        here to choose which link goes to. There are three conditions for the link based on node distance. 0,1,2
        '''
        neighbors = graph.neighbors(direction_node)
        # print G.has_edge(1,3)
        # print G.has_edge(3,1)
        '''
        calculate sum of distance, with +1 normalization
        '''
        distance_sum = 0
        for neighbor in neighbors:
            # print "neighbors:", neighbor
            neighbor_link = graph[direction_node][neighbor]  # get candidate link's type
            # print "neighbor_link: ",neighbor_link
            neighbor_link_type = neighbor_link['type']
            # print "neighbor_link_type: ",neighbor_link_type
            neighbor_link_weight = neighbor_link['weight']
            trans_weight = matrix[cur_edge_type - 1][neighbor_link_type - 1]
            if graph.has_edge(neighbor, left_node) or graph.has_edge(left_node, neighbor):
                distance_sum += trans_weight * neighbor_link_weight / p
            elif neighbor == left_node:  # decide whether it can random walk back
                distance_sum += trans_weight * neighbor_link_weight
            else:
                distance_sum += trans_weight * neighbor_link_weight / q

        '''
        pick up the next step link
        '''
        # random.shuffle(neighbors)
        rand = np.random.rand() * distance_sum
        threshold = 0
        # next_link_end_node = 0 
        neighbors2 = graph.neighbors(direction_node)
        for neighbor in neighbors2:
            # print "current threshold: ", threshold
            neighbor_link = graph[direction_node][neighbor]  # get candidate link's type
            neighbor_link_type = neighbor_link['type']
            neighbor_link_weight = neighbor_link['weight']
            trans_weight = matrix[cur_edge_type - 1][neighbor_link_type - 1]
            if graph.has_edge(neighbor, left_node) or graph.has_edge(left_node, neighbor):
                threshold += trans_weight * neighbor_link_weight / p
                if threshold >= rand:
                    next_link_end_node = neighbor
                    break
            elif neighbor == left_node:
                threshold += trans_weight * neighbor_link_weight
                if threshold >= rand:
                    next_link_end_node = neighbor
                    break
            else:
                threshold += trans_weight * neighbor_link_weight / q
                if threshold >= rand:
                    next_link_end_node = neighbor
                    break

        # print "distance_sum: ",distance_sum
        # print "rand: ", rand, " threshold: ", threshold
        # print "next_link_end_node: ",next_link_end_node

        if distance_sum > 0:  # the direction_node has next_link_end_node
            next_link = graph[direction_node][next_link_end_node]
            # next_link = G.get_edge_data(direction_node,next_link_end_node)

            next_link_tuple = tuple()
            next_link_tuple += (direction_node,)
            next_link_tuple += (next_link_end_node,)
            next_link_tuple += (next_link,)
            # print type(next_link_tuple)
            # print next_link_tuple
            walk.append(next_link_tuple)
            result.append(get_type_from_link(next_link_tuple))
            # print "walk length: ",len(walk),walk
        else:
            break
    # print "path: ",result
    return result


def get_type_from_link(edge: Edge) -> int:
    return edge[2]['type']


def update_trans_matrix(walks: Iterable[Walk], number_edge_types: int, evaluation_test) -> np.ndarray:
    """E step, update transition matrix."""
    edge_walk_vectors = defaultdict(list)

    for walk in walks:
        # Count each edge type in this walk
        edge_type_counts = Counter(
            edge - 1  # fix the off-by-one
            for edge in walk
        )

        for edge_type_id in range(number_edge_types):
            edge_walk_vectors[edge_type_id].append(edge_type_counts[edge_type_id])

    rv = np.zeros(shape=(number_edge_types, number_edge_types))

    it = tqdm(
        itertools.combinations(range(number_edge_types), r=2),
        total=number_edge_types ** 2,
        leave=False,
    )
    for i, j in it:
        if edge_walk_vectors[i] == edge_walk_vectors[j]:
            break
        else:
            rv[j][i] = rv[i][j] = evaluation_test(edge_walk_vectors[i], edge_walk_vectors[j])

    return rv
