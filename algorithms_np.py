import math

from graph_numpy import *


def two_approximate_vertex_cover(graph: GraphNP) -> List[int]:
    """
    This function finds a vertex cover using the 2-approximation algorithm
    :return: a list of vertices that are a vertex cover
    """
    vertex_cover = []
    edges = graph.get_neighbours_mtx()
    remaining_edges = edges
    arr = np.argwhere(remaining_edges)
    while arr.size > 0:
        edge = arr[np.random.randint(arr.shape[0])]
        remaining_edges[edge[0], :] = 0
        remaining_edges[edge[1], :] = 0
        remaining_edges[:, edge[0]] = 0
        remaining_edges[:, edge[1]] = 0
        vertex_cover.append(edge[0])
        vertex_cover.append(edge[1])
        arr = np.argwhere(remaining_edges)
    return vertex_cover
