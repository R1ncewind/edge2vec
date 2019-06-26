# -*- coding: utf-8 -*-

"""A major refactoring of ``edge2vec``.

A high level overview:

.. code-block:: python

    from edge2vec import calculate_edge_transition_matrix, train, read_graph
    graph = read_graph(...)
    transition_matrix = calculate_edge_transition_matrix(graph=graph, ...)
    word2vec = train(graph=graph, transition_matrix=transition_matrix, ...)
"""

from .edge2vec import train
from .transition import calculate_edge_transition_matrix
from .utils import read_graph
