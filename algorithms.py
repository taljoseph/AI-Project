import copy
import math

from numpy import mat
from genetic_algorithms import *
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


def ghc_weighted(graph: Graph, initial_state: List[int], num_iters) -> List[int]:
    edges = set(graph.get_edges())
    weights = {edge: 1 for edge in edges}
    best_cover = graph.get_vertices()
    neighbours = graph.get_neighbors()
    for i in range(num_iters):
        remaining_vertices = set(graph.get_vertices())
        remaining_edges = copy.deepcopy(edges)
        cur_state = set()
        while remaining_edges:
            cur_best_vertex = None
            cur_best_val = 0
            for vertex in remaining_vertices:
                vertex_val = 0
                z = neighbours[vertex]
                for neighbour in z:
                    if neighbour in remaining_vertices:
                        vertex_val += weights[frozenset({vertex, neighbour})]
                if vertex_val > cur_best_val:
                    cur_best_val = vertex_val
                    cur_best_vertex = vertex
            cur_state.add(cur_best_vertex)
            remaining_vertices.remove(cur_best_vertex)
            remaining_edges -= {frozenset({cur_best_vertex, u}) for u in neighbours[cur_best_vertex]}
            for edge in remaining_edges:
                weights[edge] += 1

        # removing neighbors
        valid_neighbor = True
        while valid_neighbor:
            worst_v = -1
            least_edges = math.inf
            valid_neighbor = False
            for v in cur_state:
                v_neighbours = neighbours[v]
                num_edges_covered = len(v_neighbours)
                if num_edges_covered < least_edges and not v_neighbours.difference(cur_state):
                    worst_v = v
                    least_edges = num_edges_covered
                    valid_neighbor = True
            if valid_neighbor:
                cur_state.remove(worst_v)
        if len(best_cover) > len(cur_state):
            best_cover = cur_state
    return list(best_cover)


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
        edges_covered |= get_edges_covered_by_vertex(best_vertex, neighbors_dict)

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
        edges_covered |= get_edges_covered_by_vertex(rand_vertex, neighbors_dict)

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
                edges_covered |= get_edges_covered_by_vertex(v, neighbors_dict)
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


def is_vc(graph: Graph, state: List[int]) -> bool:
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


def get_edges_covered_by_vertex(v: int, neighbors: Dict[int, Set[int]]) -> Set[FrozenSet[int]]:
    """
    This function returns a set of all edges covered by vertex v
    """
    edges_covered = set()
    for u in neighbors[v]:
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


def simulated_annealing(graph: Graph, initial_state: List[int], schedule):
    cur_state = set(initial_state)
    neighbors_dict = graph.get_neighbors()
    edges_covered = get_edges_covered(initial_state, graph)
    num_edges = graph.get_num_edges()
    val = len(edges_covered) * 2 - len(cur_state)
    t = 1
    best_sol = None
    best_sol_num_vertices = math.inf
    num_vertices = graph.get_num_vertices()

    while True:
        T = schedule(t)
        if T <= 0: # 1e-10:
            return list(cur_state) if best_sol is None else list(best_sol)
        if t % 10000 == 0:
            print(t)
        k = random.randint(0, num_vertices - 1)
        if k in cur_state:
            new_edges_covered = edges_covered.difference({frozenset({k, v}) for v in neighbors_dict[k] if v not in cur_state})
            new_val = len(new_edges_covered) * 2 - (len(cur_state) - 1)
            p = 1
            if len(new_edges_covered) < len(edges_covered):
                p = math.e ** ((new_val - val)/T)
            if random.random() < p:
                cur_state.remove(k)
                edges_covered = new_edges_covered
                val = new_val

        else:
            new_edges_covered = edges_covered | (get_edges_covered_by_vertex(k, neighbors_dict))
            new_val = len(new_edges_covered) * 2 - len(cur_state) - 1
            p = 1
            if len(new_edges_covered) == len(edges_covered):
                p = math.e ** ((new_val - val)/T)
            if random.random() < p:
                cur_state.add(k)
                edges_covered = new_edges_covered
                val = new_val
        if len(edges_covered) == num_edges and \
                len(cur_state) < best_sol_num_vertices:
            best_sol = cur_state
        t += 1


def local_beam_search(graph: Graph, k):
    k_states = []
    vertices = set(graph.get_vertices())
    for i in range(k):
        rand_state = create_random_initial_state(graph)
        edges_covered = get_edges_covered(rand_state, graph)
        value = 2 * len(edges_covered) - len(rand_state)
        rand_state = set(rand_state)
        remaining_vertices = vertices.difference(rand_state)
        k_states.append([rand_state, remaining_vertices, edges_covered, value])
    k_states = sorted(k_states, key=lambda x: x[3], reverse=True)

    num_edges = graph.get_num_edges()
    neighbours_dict = graph.get_neighbors()
    while True:
        all_neighbours = []
        any_better_neighbour = False
        for state in k_states:  # [0] = vertex cover, [1] = remaining vertices, [2] = edges_covered, [3] = value
            if len(state[2]) < num_edges:
                for vertex in state[1]:
                    new_vertices = state[0] | {vertex}
                    vertex_edges = get_edges_covered_by_vertex(vertex, neighbours_dict)
                    new_edges = state[2] | vertex_edges
                    new_value = len(new_edges) * 2 - len(new_vertices)
                    if new_value > state[3]:
                        any_better_neighbour = True
                        all_neighbours.append([new_vertices, state[1] - {vertex}, new_edges, new_value])

            else:  # num_edges = num edges covered
                all_neighbours.append(state)
                for vertex in state[0]:
                    is_valid = True
                    for neighbour in neighbours_dict[vertex]:
                        if neighbour not in state[0]:
                            is_valid = False
                            break
                    if not is_valid:
                        continue
                    new_vertices = state[0] - {vertex}
                    new_value = state[3] + 1
                    any_better_neighbour = True
                    all_neighbours.append([new_vertices, state[1] - {vertex}, state[2], new_value])
        if not any_better_neighbour:
            return list(k_states[0][0])
        all_neighbours = sorted(all_neighbours, key=lambda x: x[3], reverse=True)
        if len(all_neighbours) < k:
            k_states = all_neighbours
        else:
            k_states = all_neighbours[:k]


def ga_vc(ga: FixedSizeVCGA, graph: Graph, num_generations: int, population_size: int):
    two_approx = two_approximate_vertex_cover(graph)
    min_vc_size = round(len(two_approx) / 2)
    ghc = greedy_hill_climbing(graph, [])
    max_vc_size = min(len(two_approx), len(ghc))
    # best_val = max_vc_size
    best_sol = np.array(ghc if len(two_approx) > len(ghc) else two_approx)

    while max_vc_size > min_vc_size:
        k = (max_vc_size + min_vc_size) // 2
        print(max_vc_size, min_vc_size, k)
        ga.update_k(k)
        sol = ga.perform_ga(num_generations, population_size)
        if is_vc(graph, sol):
            best_sol = sol
            max_vc_size = k
        else:
            sol = np.array(greedy_hill_climbing(graph, sol))
            print(len(sol))
            if len(sol) < max_vc_size:
                max_vc_size = len(sol)
                best_sol = sol
            min_vc_size = k + 1
    return best_sol.tolist()


def greedy_neighbor(graph: Graph) -> List[int]:
    neighbors = graph.get_neighbors()
    vertex_degree = {k: len(v) for k, v in neighbors.items()}
    edges_covered = set()
    cur_state = set()
    num_edges = graph.get_num_edges()
    while len(edges_covered) < num_edges:
        one_deg_v = [k for k, v in vertex_degree.items() if v == 1]
        while one_deg_v:
            rand_v = random.choice(one_deg_v)
            vertex_degree[rand_v] = 0

            n1 = neighbors[rand_v].pop()
            edges_covered.add(frozenset({rand_v, n1}))
            cur_state.add(n1)
            n1_neighbors = neighbors[n1]
            for n2 in n1_neighbors:
                edges_covered.add(frozenset({n1, n2}))
                vertex_degree[n2] -= 1
                neighbors[n2] -= {n1}

            # since we added n1 to VC, we need to make it's degree is zero in order to not randomly choose it in the
            # next iteration
            vertex_degree[n1] = 0

            one_deg_v = [k for k, v in vertex_degree.items() if v == 1]

        min_two_deg_v = [k for k, v in vertex_degree.items() if v >= 2]
        if min_two_deg_v:
            rand_v = random.choice(min_two_deg_v)
            rand_v_neighbors = neighbors[rand_v]
            for n1 in rand_v_neighbors:
                edges_covered.add(frozenset({rand_v, n1}))
                cur_state.add(n1)
                neighbors[n1] -= {rand_v}
                n1_neighbors = neighbors[n1]
                vertex_degree[n1] = 0

                for n2 in n1_neighbors:
                    edges_covered.add(frozenset({n1, n2}))
                    vertex_degree[n2] -= 1
                    neighbors[n2] -= {n1}

            vertex_degree[rand_v] = 0

    return list(cur_state)
