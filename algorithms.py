from Graph import Graph


def two_approximate_vertex_cover(graph: Graph):
    """
    This function finds a vertex cover using the 2-approximation algorithm
    :return: a list of vertices that are a vertex cover
    """
    vertex_cover = []
    remaining_edges = set(graph.get_edges())
    neighbors = graph.get_neighbors()
    while remaining_edges:
        edge = list(remaining_edges.pop())
        vertex1, vertex2 = edge[0], edge[0]
        vertex_cover.append(vertex1)
        vertex_cover.append(vertex2)
        edges_to_remove = {{vertex1, v} for v in neighbors[vertex1]}
        edges_to_remove |= {{vertex2, v} for v in neighbors[vertex2]}
        remaining_edges -= edges_to_remove
    return vertex_cover
