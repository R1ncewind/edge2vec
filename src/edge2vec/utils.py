import networkx as nx

__all__ = [
    'read_graph',
]


def read_graph(path: str, weighted: bool = False, directed: bool = False):
    """Reads the input network in networkx."""
    create_using = (nx.DiGraph if directed else nx.Graph)

    if weighted:
        g = nx.read_edgelist(
            path,
            nodetype=str,
            data=(('type', int), ('weight', float), ('id', int)),
            create_using=create_using,
        )
    else:
        g = nx.read_edgelist(
            path,
            nodetype=str,
            data=(('type', int), ('id', int)),
            create_using=create_using,
        )
        for source, target in g.edges():
            g[source][target]['weight'] = 1.0

    return g
