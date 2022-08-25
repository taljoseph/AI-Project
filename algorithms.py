import copy
from utils import *


def two_approximate_vertex_cover(graph: Graph) -> List[int]:
    """
    This function finds a vertex cover using the 2-approximation algorithm
    @param graph: A graph object
    @return: a list of vertices representing a vertex cover
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
    This function finds a vertex cover using the greedy hill climbing algorithm which operates similarly to
    using a cost function of (2+eps) * |E'| - sum(|V|-deg(v))/|V|) over each v in V'.
    @param graph: A graph object.
    @param initial_state: The initial state.
    @return: a list of vertices representing a vertex cover.
    """
    # init
    cur_state = set(initial_state)
    remaining_vertices_set = set(graph.get_vertices()).difference(cur_state)
    neighbors_dict = graph.get_neighbors()

    # adding neighbours
    edges_covered = get_edges_covered(initial_state, graph)
    num_edges = graph.get_num_edges()
    while len(edges_covered) < num_edges:
        best_vertices = []
        most_edges_added = -1
        for v in remaining_vertices_set:
            new_edges_added = len(neighbors_dict[v].difference(cur_state))
            if new_edges_added > most_edges_added:
                best_vertices = [v]
                most_edges_added = new_edges_added
            elif new_edges_added == most_edges_added:
                best_vertices.append(v)
        rand_vertex = random.choice(best_vertices)
        cur_state.add(rand_vertex)
        remaining_vertices_set.remove(rand_vertex)
        edges_covered |= get_edges_covered_by_vertex(rand_vertex, neighbors_dict)

    # removing neighbours
    valid_neighbor = True
    while valid_neighbor:
        worst_vertices = []
        least_edges = math.inf
        valid_neighbor = False
        for v in cur_state:
            v_neighbors = neighbors_dict[v]
            num_edges_covered = len(v_neighbors)
            if num_edges_covered <= least_edges and not v_neighbors.difference(cur_state):
                if num_edges_covered == least_edges:
                    worst_vertices.append(v)
                else:
                    worst_vertices = [v]
                    least_edges = num_edges_covered
                valid_neighbor = True
        if valid_neighbor:
            rand_vertex = random.choice(worst_vertices)
            cur_state.remove(rand_vertex)
    return list(cur_state)


def stochastic_hill_climbing(graph: Graph, initial_state: List[int]) -> List[int]:
    """
    This function finds a vertex cover using the stochastic hill climbing algorithm, which works similarly
    to using the cost function above.
    @param graph: A graph object.
    @param initial_state: The initial state.
    @return: a list of vertices representing a vertex cover.
    """
    # init
    cur_state = set(initial_state)
    rest_vertices_set = set(graph.get_vertices()).difference(cur_state)
    neighbors_dict = graph.get_neighbors()

    # adding neighbours
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

    # removing neighbours
    valid_neighbor = True
    while valid_neighbor:
        worst_vertices = []
        valid_neighbor = False
        for v in cur_state:
            v_neighbors = neighbors_dict[v]
            if not v_neighbors.difference(cur_state):
                worst_vertices.append(v)
                valid_neighbor = True
        if valid_neighbor:
            cur_state.remove(random.choice(worst_vertices))
    return list(cur_state)


def random_restart_hill_climbing(graph: Graph, num_iters: int, algorithm) -> List[int]:
    """
    Finds a vertex cover using random restart hill climbing algorithm
    @param graph: The graph object.
    @param num_iters: The number of iterations.
    @param algorithm: Algorithm used with random restart.
    @return: A list of vertices representing a vertex cover.
    """
    best_vc = []
    len_of_best = math.inf
    for i in range(num_iters):
        init_state = create_random_initial_state(graph)
        cur_vc = algorithm(graph, init_state)
        if len(cur_vc) < len_of_best:
            best_vc = cur_vc
            len_of_best = len(cur_vc)
    return best_vc


def ghc_weighted_edges(graph: Graph, num_iters) -> List[int]:
    """
    Calculates weights on edges, increasing the weight for each edge that isn't covered in the following round
    and runs Hill Climbing with the cost function being the sum of the weights of the relevant edges on each vertex.
    @param graph: The graph object
    @param num_iters: Number of iterations to run this algorithm, keeping the best vertex cover found so far
    @return: The best vertex found thus far.
    """
    # init
    edges = set(graph.get_edges())
    weights = {edge: 1 for edge in edges}
    best_cover = graph.get_vertices()
    neighbours = graph.get_neighbors()
    for i in range(num_iters):
        remaining_vertices = set(graph.get_vertices())
        remaining_edges = copy.deepcopy(edges)
        cur_state = set()

        # adding neighbours
        while remaining_edges:
            cur_best_vertices = []
            cur_best_val = 0
            for vertex in remaining_vertices:
                vertex_val = 0
                z = neighbours[vertex]
                for neighbour in z:
                    if neighbour in remaining_vertices:
                        vertex_val += weights[frozenset({vertex, neighbour})]
                if vertex_val > cur_best_val:
                    cur_best_val = vertex_val
                    cur_best_vertices = [vertex]
                elif vertex_val == cur_best_val:
                    cur_best_vertices.append(vertex)
            rand_vertex = random.choice(cur_best_vertices)
            cur_state.add(rand_vertex)
            remaining_vertices.remove(rand_vertex)
            remaining_edges -= {frozenset({rand_vertex, u}) for u in neighbours[rand_vertex]}
            for edge in remaining_edges:
                weights[edge] += 1

        # removing neighbours
        valid_neighbor = True
        while valid_neighbor:
            worst_vertices = []
            least_edges = math.inf
            valid_neighbor = False
            for v in cur_state:
                v_neighbours = neighbours[v]
                num_edges_covered = len(v_neighbours)
                if num_edges_covered <= least_edges and not v_neighbours.difference(cur_state):
                    if num_edges_covered == least_edges:
                        worst_vertices.append(v)
                    else:
                        least_edges = num_edges_covered
                        worst_vertices = [v]
                    valid_neighbor = True
            if valid_neighbor:
                rand_vertex = random.choice(worst_vertices)
                cur_state.remove(rand_vertex)
        if len(best_cover) > len(cur_state):
            best_cover = cur_state
    return list(best_cover)


def ghc_weighted_vertices(graph: Graph, initial_state: List[int]):
    """
    Calculates weights on the vertices based on a certain function and runs
    Hill Climbing with the cost function being the sum of the weights of the relevant vertices.
    @param graph: The graph object.
    @param initial_state: The beginning.
    @return: The best vertex found thus far.
    """
    state = set(initial_state)
    neighbours = graph.get_neighbors()
    edges = set(graph.get_edges())
    remaining_edges = copy.deepcopy(edges) - get_edges_covered(initial_state, graph)
    remaining_vertices = set(graph.get_vertices()) - state
    weights = {vertex: 0 for vertex in remaining_vertices}
    remaining_neighbours = {vertex: set() for vertex in remaining_vertices}
    for edge in remaining_edges:
        v1, v2 = edge
        remaining_neighbours[v1].add(v2)
        remaining_neighbours[v2].add(v1)

    while remaining_edges:
        precious_vertex = False
        for vertex in remaining_vertices:
            weights[vertex] = 0
        for vertex in remaining_vertices:
            if len(remaining_neighbours[vertex]) == 1:
                v, = remaining_neighbours[vertex]
                weights[v] = math.inf
                precious_vertex = True
        if not precious_vertex:
            for vertex in remaining_vertices:
                max_val = -math.inf
                min_val = math.inf
                for neighbour in remaining_neighbours[vertex]:
                    num_rel_neighbours = len(remaining_neighbours[neighbour])
                    if num_rel_neighbours > max_val:
                        max_val = num_rel_neighbours
                    elif num_rel_neighbours < min_val:
                        min_val = num_rel_neighbours
                num_neighbours = len(remaining_neighbours[vertex])
                if num_neighbours == 0:
                    continue
                elif num_neighbours >= max_val:
                    score = min_val
                else:
                    score = max_val
                for neighbour in remaining_neighbours[vertex]:
                    weights[neighbour] += score
        max_val = max(weights.values())
        best_vertices = [k for k, v in weights.items() if v == max_val]
        best_vertex = random.choice(best_vertices)
        state.add(best_vertex)
        weights[best_vertex] = 0
        for neighbour in remaining_neighbours[best_vertex]:
            remaining_neighbours[neighbour].remove(best_vertex)
            remaining_edges.remove(frozenset({best_vertex, neighbour}))
        del remaining_neighbours[best_vertex]
        remaining_vertices.remove(best_vertex)

    # Reduce vertex cover

    weights = {}
    neighbours_in_state = {vertex: set() for vertex in state}
    for vertex in state:
        neighbours_in_state[vertex] = neighbours[vertex].intersection(state)

    can_reduce_cover = True
    while can_reduce_cover:
        for vertex in state:
            cant_remove_vertex = False
            for neighbour in neighbours[vertex]:
                if neighbour not in state:
                    cant_remove_vertex = True
                    break
            if cant_remove_vertex:
                continue
            # can remove vertex, set its weights:
            num_neighbours = len(neighbours_in_state[vertex])
            if num_neighbours == 0:
                weights[vertex] = math.inf
                break

            score = 0
            for neighbour in neighbours_in_state[vertex]:
                max_val = -math.inf
                min_val = math.inf
                for n_o_n in neighbours_in_state[neighbour]:
                    num_rel_neighbours = len(neighbours_in_state[n_o_n])
                    if num_rel_neighbours > max_val:
                        max_val = num_rel_neighbours
                    elif num_rel_neighbours < min_val:
                        min_val = num_rel_neighbours
                if num_neighbours >= max_val:
                    score += num_neighbours
                else:
                    score += min_val
            weights[vertex] = score

        if not weights:
            can_reduce_cover = False
        else:
            best_vertex = max(weights, key=weights.get)
            state.remove(best_vertex)
            del weights[best_vertex]
            for neighbour in neighbours_in_state[best_vertex]:
                neighbours_in_state[neighbour].remove(best_vertex)
            del neighbours_in_state[best_vertex]
            weights = {}
    return state


def cost4(num_edges_covered, num_vertices_state, num_edges):
    """
    Cost function number 4: 2*|E'|-|V'| where E' = edges covered, V' = vertices in cover.
    @param num_edges_covered: Number of edges covered by the vertices
    @param num_vertices_state: number of the vertices in the state (in the vc)
    @param num_edges: number of edges in the graph
    @return: Value of the cost function as explained above
    """
    return 2 * num_edges_covered - num_vertices_state


def cost5(num_edges_covered, num_vertices_state, num_edges):
    """
    Cost function number 5: |E'|-|V'| where E' = edges covered, V' = vertices in cover
    @param num_edges_covered: Number of edges covered by the vertices
    @param num_vertices_state: number of the vertices in the state (in the vc)
    @param num_edges: number of edges in the graph
    @return: Value of the cost function as explained above
    """
    return num_edges_covered - num_vertices_state


def cost6(num_edges_covered, num_vertices_state, num_edges):
    """
    Cost function number 6: |E'|-|V'| - punishment, where E' = edges covered, V' = vertices in cover.
    @param num_edges_covered: Number of edges covered by the vertices
    @param num_vertices_state: number of the vertices in the state (in the vc)
    @param num_edges: number of edges in the graph
    @return: Value of the cost function as explained above
    """
    punishment = 5 if num_edges > num_edges_covered else 0
    return num_edges_covered - num_vertices_state - punishment


def simulated_annealing(graph: Graph, initial_state: List[int], schedule, cost):
    """
    This function finds a vertex cover using Simulated Annealing algorithm with the given schedule.
    Given the nature of this algorithm, a VC isn't promised.
    @param graph: The graph object
    @param initial_state: The beginning state, usually empty
    @param schedule: The schedule, i.e. temperature function
    @param cost: The cost function
    @return: The state the algorithm reached, hopefully reaching a VC.
    """
    cur_state = set(initial_state)
    neighbors_dict = graph.get_neighbors()
    edges_covered = get_edges_covered(initial_state, graph)
    num_edges = graph.get_num_edges()
    cur_state_val = cost(len(edges_covered), len(cur_state), num_edges)
    t = 1
    best_sol = None
    best_sol_num_vertices = math.inf
    num_vertices = graph.get_num_vertices()

    while True:
        T = schedule(t)
        if T <= 0:
            return list(cur_state) if best_sol is None else list(best_sol)
        k = random.randint(0, num_vertices - 1)
        if k in cur_state:
            new_edges_covered = edges_covered.difference({frozenset({k, v}) for v in neighbors_dict[k] if v not in cur_state})
            new_val = cost(len(new_edges_covered), len(cur_state) - 1, num_edges)
            p = 1
            if new_val < cur_state_val:
                p = math.e ** ((new_val - cur_state_val)/T)
            if random.random() < p:
                cur_state.remove(k)
                edges_covered = new_edges_covered
                cur_state_val = new_val

        else:
            new_edges_covered = edges_covered | (get_edges_covered_by_vertex(k, neighbors_dict))
            new_val = cost(len(new_edges_covered), len(cur_state) + 1, num_edges)
            p = 1
            if new_val < cur_state_val:
                p = math.e ** ((new_val - cur_state_val)/T)
            if random.random() < p:
                cur_state.add(k)
                edges_covered = new_edges_covered
                cur_state_val = new_val
        if len(edges_covered) == num_edges and \
                len(cur_state) < best_sol_num_vertices:
            best_sol = cur_state
        t += 1


def local_beam_search(graph: Graph, k):
    """
    This function finds a vertex cover using Local Beam Search algorithm with k best states
    @param graph: The graph object
    @param k:
    @return:
    """
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


########################################################################################
#  EXTRAS ##############################################################################
########################################################################################

def ghc_weighted_edges_2(graph: Graph, initial_state: List[int]):
    """
    Calculates weights on edges based on a certain function and runs
    Hill Climbing with the cost function being the sum of the weights of the relevant edges on each vertex.
    @param graph: The graph object.
    @param initial_state: The beginning.
    @return: The best vertex found thus far.
    """
    # init
    state = set(initial_state)
    neighbours = graph.get_neighbors()
    edges = set(graph.get_edges())
    remaining_edges = copy.deepcopy(edges) - get_edges_covered(initial_state, graph)
    remaining_vertices = set(graph.get_vertices()) - state
    weights = {edge: 0 for edge in edges}

    # set weights, calculated by number of neighbours not covered each vertex has (less is more):
    for edge in remaining_edges:
        for vertex in edge:
            num_non_covered_neighbours = 0
            for neighbour in neighbours[vertex]:
                if neighbour in remaining_vertices:  # meaning neighbour not covered
                    num_non_covered_neighbours += 1
            if num_non_covered_neighbours == 1:
                weights[edge] += graph.get_num_vertices() ** 2
            else:
                weights[edge] += graph.get_num_vertices() ** (1 / num_non_covered_neighbours)

    while remaining_edges:
        cur_best_score = 0
        cur_best_vertex = None
        for vertex in remaining_vertices:
            vertex_score = 0
            for neighbour in neighbours[vertex]:
                if neighbour in remaining_vertices:  # meaning the edge not covered
                    vertex_score += weights[frozenset({vertex, neighbour})]
            if vertex_score > cur_best_score:
                cur_best_score = vertex_score
                cur_best_vertex = vertex
        if cur_best_vertex is not None:
            state.add(cur_best_vertex)
            remaining_vertices.remove(cur_best_vertex)
            remaining_edges -= {frozenset({cur_best_vertex, n}) for n in neighbours[cur_best_vertex]}

    # Reduce vertex cover
    for edge in weights:
        weights[edge] = 0
        for vertex in edge:
            if len(neighbours[vertex]) == 1:
                weights[edge] -= graph.get_num_vertices() ** 2
            else:
                weights[edge] += graph.get_num_vertices() ** (1 / len(neighbours[vertex]))

    can_reduce_cover = True
    while can_reduce_cover:
        cur_best_score = math.inf
        cur_best_vertex = None
        for vertex in state:
            cur_score = 0
            cant_remove_vertex = False
            for neighbour in neighbours[vertex]:
                if neighbour not in state:
                    cant_remove_vertex = True
                    break
                cur_score += weights[frozenset({vertex, neighbour})]
            if cant_remove_vertex:
                continue
            if cur_score < cur_best_score:
                cur_best_score = cur_score
                cur_best_vertex = vertex
        if not cur_best_vertex:
            can_reduce_cover = False
        else:
            state.remove(cur_best_vertex)
    return state


def first_choice_hill_climbing(graph: Graph, initial_state: List[int]) -> List[int]:
    """
    Finds a vertex cover using first choice hill climbing algorithm, works similarly to using the cost function
    above.
    @param graph: The graph object.
    @param initial_state: The initial state.
    @return: A list of vertices representing a vertex cover.
    """
    # init
    cur_state = set(initial_state)
    rest_vertices_set = set(graph.get_vertices()).difference(cur_state)
    neighbors_dict = graph.get_neighbors()

    # adding neighbours
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

    # removing neighbours
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


def multirun_whc_weighted_edges_2(graph: Graph, num_iters: int) -> List[int]:
    """
    Runs whc_weighted_edged_2 multiple times.
    @param graph: Graph object.
    @param num_iters: Number of iterations.
    @return: The best vertex cover found from all iterations.
    """
    best_vc = []
    len_of_best = math.inf
    for i in range(num_iters):
        init_state = create_random_initial_state(graph)
        cur_vc = ghc_weighted_edges_2(graph, init_state)
        if len(cur_vc) < len_of_best:
            best_vc = cur_vc
            len_of_best = len(cur_vc)
    return best_vc


def multirun_whc_weighted_vertices(graph: Graph, num_iters: int) -> List[int]:
    """
    Runs whc_weighted_vertices multiple times.
    @param graph: Graph object.
    @param num_iters: Number of iterations.
    @return: The best vertex cover found from all iterations.
    """
    best_vc = []
    len_of_best = math.inf
    for i in range(num_iters):
        cur_vc = ghc_weighted_vertices(graph, [])
        if len(cur_vc) < len_of_best:
            best_vc = cur_vc
            len_of_best = len(cur_vc)
    return best_vc


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
