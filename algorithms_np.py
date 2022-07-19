import math

from graph_numpy import *


def two_approximate_vertex_cover(graph: GraphNP) -> List[int]:
    """
    This function finds a vertex cover using the 2-approximation algorithm
    :return: a list of vertices that are a vertex cover
    """
    vertex_cover = []
    remaining_edges = set(graph.get_edges())
    neighbors = graph.get_neighbors()
    while remaining_edges:
        edge = list(remaining_edges.pop())
        vertex1, vertex2 = edge[0], edge[1]
        vertex_cover.append(vertex1)
        vertex_cover.append(vertex2)
        edges_to_remove = {frozenset({vertex1, v}) for v in neighbors[vertex1]}
        edges_to_remove |= {frozenset({vertex2, v}) for v in neighbors[vertex2]}
        remaining_edges -= edges_to_remove
    return vertex_cover


def greedy_hill_climbing(graph: Graph, initial_state: List[int]) -> List[int]:
    """
    This function finds a vertex cover using the greedy hill climbing algorithm
    return: a list of vertices that are a vertex cover
    """
    cur_state = initial_state
    #adding neighbors
    edges_covered = get_edges_covered(initial_state, graph)
    num_edges = graph.get_num_edges()
    while len(edges_covered) < num_edges:
        best_vertex = -1
        most_edges_added = -1
        for v in list(set(graph.get_vertices()).difference(set(cur_state))):
            new_edges_added = len(graph.get_neighbors()[v].difference(set(cur_state)))
            if new_edges_added > most_edges_added:
                best_vertex = v
                most_edges_added = new_edges_added
        cur_state.append(best_vertex)
        edges_covered |= get_edges_covered_by_vertex(best_vertex, graph)

    # removing neighbors
    valid_neighbor = True
    while valid_neighbor:
        worst_v = -1
        least_edges = math.inf
        valid_neighbor = False
        for i in range(len(cur_state)):
            v = cur_state[i]
            v_neighbors = graph.get_neighbors()[v]
            num_edges_covered = len(v_neighbors)
            if not v_neighbors.difference(set(cur_state)) and num_edges_covered < least_edges:
                worst_v = v
                least_edges = num_edges_covered
                valid_neighbor = True
        if valid_neighbor:
            cur_state.remove(worst_v)

    return cur_state


def is_goal_state(graph: Graph, state: List[int]) -> bool:
    """
    This function checks if the state is the goal state - a vertex cover
    return: True if goal state, otherwise false
    """
    edges = graph.get_edges()
    vc = set(state)
    for edge in edges:
        e = list(edge)
        if e[0] not in vc and e[1] not in vc:
            return False
    return True


def get_edges_covered_by_vertex(v: int, graph: Graph) -> Set[FrozenSet[int]]:
    """
    This function returns a set of all edges covered by vertex v
    """
    edges_covered = set()
    for u in graph.get_neighbors()[v]:
        edges_covered.add(frozenset({u, v}))
    return edges_covered


def get_edges_covered(state: List[int], graph: Graph) -> Set[FrozenSet[int]]:
    """
    This function returns a set of all edges covered by the current state
    """
    edges_covered = set()
    for u in state:
        for u_neighbor in graph.get_neighbors()[u]:
            edges_covered.add(frozenset({u, u_neighbor}))
    return edges_covered
