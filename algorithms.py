import math

from numpy import mat

from Graph import *


def two_approximate_vertex_cover(graph: Graph) -> List[int]:
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
    cur_state = set(initial_state)
    rest_vertices_set = set(graph.get_vertices()).difference(cur_state)
    neighbors_dict = graph.get_neighbors()
    #adding neighbors
    edges_covered = get_edges_covered(initial_state, graph)
    num_edges = graph.get_num_edges()
    while len(edges_covered) < num_edges:
        best_vertex = -1
        most_edges_added = -1
        for v in rest_vertices_set:
            new_edges_added = len(neighbors_dict[v].difference(cur_state))
            if new_edges_added > most_edges_added:
                best_vertex = v
                most_edges_added = new_edges_added
        cur_state.add(best_vertex)
        rest_vertices_set.remove(best_vertex)
        edges_covered |= get_edges_covered_by_vertex(best_vertex, graph)

    # removing neighbors
    valid_neighbor = True
    while valid_neighbor:
        worst_v = -1
        least_edges = math.inf
        valid_neighbor = False
        for v in cur_state:
            v_neighbors = neighbors_dict[v]
            num_edges_covered = len(v_neighbors)
            if num_edges_covered < least_edges and not v_neighbors.difference(cur_state):
                worst_v = v
                least_edges = num_edges_covered
                valid_neighbor = True
        if valid_neighbor:
            cur_state.remove(worst_v)
    return list(cur_state)

def stochastic_hill_climbing(graph: Graph, initial_state: List[int]) -> List[int]:
    """
    This function finds a vertex cover using the stochastic hill climbing algorithm
    return: a list of vertices that are a vertex cover
    """
    cur_state = set(initial_state)
    rest_vertices_set = set(graph.get_vertices()).difference(cur_state)
    neighbors_dict = graph.get_neighbors()
    #adding neighbors
    edges_covered = get_edges_covered(initial_state, graph)
    num_edges = graph.get_num_edges()
    while len(edges_covered) < num_edges:
        good_vertex = []
        for v in rest_vertices_set:
            new_edges_added = len(neighbors_dict[v].difference(cur_state))
            if new_edges_added:
                good_vertex.append(v)
        rand_vertex = random.choice(good_vertex)
        cur_state.add(rand_vertex)
        rest_vertices_set.remove(rand_vertex)
        edges_covered |= get_edges_covered_by_vertex(rand_vertex, graph)

    # removing neighbors
    valid_neighbor = True
    while valid_neighbor:
        worst_vertex = []
        valid_neighbor = False
        for v in cur_state:
            v_neighbors = neighbors_dict[v]
            if not v_neighbors.difference(cur_state):
                worst_vertex.append(v)
                valid_neighbor = True
        if valid_neighbor:
            cur_state.remove(random.choice(worst_vertex))
    return list(cur_state)


def first_choice_hill_climbing(graph: Graph, initial_state: List[int]) -> List[int]:
    """
    This function finds a vertex cover using the greedy hill climbing algorithm
    return: a list of vertices that are a vertex cover
    """
    cur_state = set(initial_state)
    rest_vertices_set = set(graph.get_vertices()).difference(cur_state)
    neighbors_dict = graph.get_neighbors()
    #adding neighbors
    edges_covered = get_edges_covered(initial_state, graph)
    num_edges = graph.get_num_edges()
    while len(edges_covered) < num_edges:
        for v in rest_vertices_set:
            new_edges_added = len(neighbors_dict[v].difference(cur_state))
            if new_edges_added:
                cur_state.add(v)
                rest_vertices_set.remove(v)
                edges_covered |= get_edges_covered_by_vertex(v, graph)
                break

    # removing neighbors
    valid_neighbor = True
    while valid_neighbor:
        valid_neighbor = False
        for v in cur_state:
            v_neighbors = neighbors_dict[v]
            if not v_neighbors.difference(cur_state):
                cur_state.remove(v)
                valid_neighbor = True
                break
    return list(cur_state)

def random_restart_hill_climbing(graph: Graph, num_iters: int) -> List[int]:
    init_state = []
    best_vc = []
    len_of_best = math.inf
    for i in range(num_iters):
        init_state = create_random_initial_state(graph)
        cur_vc = greedy_hill_climbing(graph, init_state)
        if len(cur_vc) < len_of_best:
            best_vc = cur_vc
            len_of_best = len(cur_vc)
    return best_vc

def create_random_initial_state(graph: Graph):
    """
    creates and returns a random initial state - random subsequence of the
    vertices in the graph
    """
    num_vertices_graph = graph.get_num_vertices()
    num_vertices_vc = random.randint(0, num_vertices_graph)
    return random.sample([i for i in range(num_vertices_graph)], num_vertices_vc)


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
