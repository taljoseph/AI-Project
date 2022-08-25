# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from algorithms import *
from genetic_algorithms import *
from vc_problem import *
import time
import glob
from enum import Enum
import xlsxwriter as xl
import pandas as pd
import openpyxl


class Algorithm(Enum):
    """
    Type of algorithm Enum
    """
    TWO_APPROX = 1  # graph
    HC = 2   # Hill Climbing: graph, initial state
    ITERATIVE = 3  # Iterative Hill Climbing: graph, num iterations
    SA = 4  # Simulated Annealing: graph, initial state, schedule function
    LBS = 5  # Local Beam Search: graph, num children
    GA = 6  # Genetic Algorithms: num gens, population size


class InitialState(Enum):
    """
    Type of starting state Enum
    """
    EMPTY = 1
    FULL = 2
    RANDOM = 3
    TWO_APPROX_COVER = 4


def create_random_initial_state(graph: Graph):
    """
    creates and returns a random initial state - random subsequence of the
    vertices in the graph
    @param graph: A Graph object
    @return: A random initial state
    """
    num_vertices_vc = random.randint(0, graph.get_num_vertices())
    return random.sample(graph.get_vertices(), num_vertices_vc)


def is_vc(graph: Graph, state: List[int]) -> bool:
    """
    This function checks if the state is the goal state - a vertex cover
    @param graph: The Graph object in question
    @param state: The state to check if it represents a goal state in the graph
    @return: True if goal state, otherwise false
    """
    edges = graph.get_edges()
    vc = set(state)
    for edge in edges:
        e = list(edge)
        if e[0] not in vc and e[1] not in vc:
            return False
    return True


def get_edges_covered_by_vertex(v: int, neighbours: Dict[int, Set[int]]) -> Set[FrozenSet[int]]:
    """
    This function returns a set of all edges covered by vertex v
    @param v: The vertex in question
    @param neighbours: The neighbours dictionary representing the neighbours of every vertex
    @return: The edges covered by vertex v
    """
    edges_covered = set()
    for u in neighbours[v]:
        edges_covered.add(frozenset({u, v}))
    return edges_covered


def get_edges_covered(state: List[int], graph: Graph) -> Set[FrozenSet[int]]:
    """
    This function returns a set of all edges covered by the current state
    @param state: The state in question
    @param graph: The graph object
    @return: A set of all edges in the graph covered in the state
    """
    edges_covered = set()
    for u in state:
        for u_neighbor in graph.get_neighbors()[u]:
            edges_covered.add(frozenset({u, u_neighbor}))
    return edges_covered


def build_graph_from_file(file_path):
    """
    Builds a graph from an mis file
    @param file_path: The path of the graph file
    @return: The graph created
    """
    file = open(file_path, "r")
    first_line = file.readline().split()
    num_of_vertices = int(first_line[2])
    edges = []
    for line in file:
        line = line.split()
        if line[0] == "e":
            edges.append((int(line[1]) - 1, int(line[2]) - 1))
    graph = Graph()
    graph.create_graph(num_of_vertices, edges)
    file.close()
    return graph


def create_random_graph_file(num_vertices, num_edges, path):
    """
    Creates an mis graph file for a random graph
    @param num_vertices: Number of vertices
    @param num_edges: Number of edges
    @param path: Path to save the file to
    """
    g = Graph()
    g.create_nx_graph(num_vertices, num_edges)
    create_graph_file(g, path)


def create_p_random_graph_file(num_vertices, p, path):
    """
    Creates an mis graph file for a p-random graph
    @param num_vertices: Number of vertices
    @param p: The probability for each edge
    @param path: Path to save the file to
    """
    g = Graph()
    g.create_p_random_graph(num_vertices, p)
    create_graph_file(g, path)


def create_graph_file(graph, path):
    """
    Creates an mis graph file
    @param graph: The graph to create a file for
    @param path: The path to save to
    """
    f = open(path, "w")
    f.write("p edge {} {}\n".format(graph.get_num_vertices(), graph.get_num_edges()))
    for edge in graph.get_edges():
        v, u = edge
        f.write("e {} {}\n".format(v + 1, u + 1))
    f.close()


def run_algorithm_on_graph(params, num_iter: int, algorithm_type: Algorithm,
                           algorithm, initial_state: InitialState = InitialState.EMPTY,
                           run_hill_climbing=False, num_vert_start=0):
    """
    Runs the requested algorithm over a graph
    @param params: The paramters required to run the algorithm
    @param num_iter: Number of iterations
    @param algorithm_type: The type of the algorithm as specified above
    @param algorithm: The algorithm function
    @param initial_state: The beginning state
    @param run_hill_climbing: Run Hill Climbing if true
    @param num_vert_start: Number of vertices in the beginning state
    @return: A list containing the algorithm's name, average vc size, minimum vc size, average run time
    """
    total_time = 0
    min_vc = math.inf
    total_vc_size = 0
    res = [algorithm.__name__]
    func = algorithm
    rel_params = params
    if algorithm_type == Algorithm.GA:
        ga = algorithm(params[0])
        func = ga.perform_ga
        rel_params = params[1:]
    for j in range(num_iter):
        if initial_state == InitialState.RANDOM:
            rel_params[1] = create_random_initial_state(params[0])
        elif initial_state == InitialState.TWO_APPROX_COVER:
            new_state = random.sample(params[0].get_vertices(), num_vert_start)
            rel_params[1] = new_state
        elif initial_state == InitialState.FULL:
            rel_params[1] = params[0].get_vertices()
        vc, run_time = get_time_and_state(func, rel_params)
        total_time += run_time
        if run_hill_climbing:
            vc, extra_time = get_time_and_state(greedy_hill_climbing, [params[0], vc])
            total_time += extra_time
        if len(vc) < min_vc:
            min_vc = len(vc)
        total_vc_size += len(vc)
    res.append(total_vc_size / num_iter)
    res.append(min_vc)
    res.append(total_time / num_iter)
    return res


def get_time_and_state(func, params):
    """
    Runs the function func with its parameters and calculates the run time
    @param func: The function to run
    @param params: The parameters relevant to the function
    @return: The vertex cover and the run time
    """
    start_time = time.time()
    vc = func(*params)
    end_time = time.time()
    return vc, end_time - start_time


def create_results_files(graph_name: str, graph, init_state: InitialState, start_vertices=0):
    """
    Runs the algorithms on the graph file and prints the results to an Excel file
    @param graph_name: The name of the graph
    @param graph: The graph object
    @param init_state: The beginning state
    @param start_vertices: Number of vertices to start with in the relevant algroithms
    """
    column_names = ["Algorithm Name", "Avg VC Size", "Min VC Size", "Avg Run Time (sec)"]
    workbook = xl.Workbook(graph_name + ".xlsx")
    worksheet = workbook.add_worksheet()
    for i in range(len(column_names)):
        worksheet.write(0, i, column_names[i])
    results = []
    two_approx_vc = run_algorithm_on_graph([graph], 10, Algorithm.TWO_APPROX, two_approximate_vertex_cover)
    if init_state == InitialState.TWO_APPROX_COVER:
        start_vertices = two_approx_vc[2] // 2
    results.append(run_algorithm_on_graph([graph, []], 10, Algorithm.HC, greedy_hill_climbing,
                                          init_state, run_hill_climbing=False, num_vert_start=start_vertices))
    results.append(run_algorithm_on_graph([graph, []], 10, Algorithm.HC, stochastic_hill_climbing,
                                          init_state, run_hill_climbing=False, num_vert_start=start_vertices))
    results.append(run_algorithm_on_graph([graph, 100], 10, Algorithm.ITERATIVE, random_restart_hill_climbing))
    results.append(run_algorithm_on_graph([graph, [], lambda x: 1 - 0.00001 * x, cost4], 10, Algorithm.SA,
                                          simulated_annealing, init_state, run_hill_climbing=True,
                                          num_vert_start=start_vertices))
    results.append(run_algorithm_on_graph([graph, 25], 10, Algorithm.LBS, local_beam_search))
    results.append(run_algorithm_on_graph([graph, 100], 1, Algorithm.ITERATIVE, ghc_weighted_edges))
    results.append(run_algorithm_on_graph([graph, []], 10, Algorithm.HC, ghc_weighted_vertices,
                                          init_state, run_hill_climbing=False, num_vert_start=start_vertices))

    results.append(run_algorithm_on_graph([graph, 10000, get_population_size(graph.get_num_vertices())], 10,
                                          Algorithm.GA, RegularVC_GA, InitialState.EMPTY, run_hill_climbing=True))
    results.append(run_algorithm_on_graph([graph, 10000, get_population_size(graph.get_num_vertices())], 10,
                                          Algorithm.GA, RegularVC_GA2, InitialState.EMPTY, run_hill_climbing=True))
    results.append(run_algorithm_on_graph([graph, 10000, get_population_size(graph.get_num_vertices())], 10,
                                          Algorithm.GA, VCPunish_GA, InitialState.EMPTY, run_hill_climbing=True))
    row = 1
    for result in results:
        for i in range(len(result)):
            worksheet.write(row, i, result[i])
        row += 1
    workbook.close()


def get_population_size(num_vertices):
    """
    Gets the recommended population size based on the number of vertices
    @param num_vertices: The number of vertices
    @return: The recommended population size
    """
    return round((0.315 * (num_vertices ** 0.6) + 0.72))


def algorithms_total_scores(path, column_name: str):
    """
    Calculates the scores of each algorithm over all results files
    @param path: The path of the directory containing the graphs
    @param column_name: The name of the column - Avg VC Size OR Min VC Size
    @return: The scores
    """
    files = glob.glob(path)
    sum_avg_values = np.zeros(4)
    for file in files:
        df = pd.read_excel(file)
        a = df[column_name]
        a = a.to_numpy()
        sum_avg_values += a
    return sum_avg_values
