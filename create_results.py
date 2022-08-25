from enum import Enum
from algorithms import *
from utils import *
import xlsxwriter as xl
import time
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