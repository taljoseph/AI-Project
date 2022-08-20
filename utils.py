# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from graph_numpy import *
from Graph import *
from algorithms import *
from genetic_algorithms import *
import random
from vc_problem import *
import time
import glob
from enum import Enum
import xlsxwriter as xl
import inspect

class Algorithm(Enum):
    TWO_APPROX = 1  # graph
    HC = 2   # Hill Climbing: graph, initial state
    ITERATIVE = 3  # Iterative Hill Climbing: graph, num iterations
    SA = 4  # Simulated Annealing: graph, initial state, schedule function
    LBS = 5  # Local Beam Search: graph, num children
    GA = 6  # Genetic Algorithms: num gens, population size

class InitialState(Enum):
    EMPTY = 1
    FULL = 2
    RANDOM = 3
    TWO_APPROX_COVER = 4


def create_plot(time_list: np.ndarray, vc_size_list: np.ndarray, is_vc_list: np.ndarray, iters: int, num_vertices: int):
    time_list_avg = time_list / iters
    vc_len_list_avg = vc_size_list / iters
    is_vc_list_avg = is_vc_list / iters

    algo_names = ["2-Approx", "ghc-weighted", "Greedy\nHill Climbing", "First Choice\nHill Climbing", "Random Restart\nHill Climbing",
                  "Stochastic\nHill Climbing", "Local Beam\nSearch", "Simulated\nAnnealing", "Regular GA", "Punish GA"]

    time_fig = plt.figure(figsize=(15, 10))
    plt.bar(algo_names, time_list_avg)
    plt.xlabel("Algorithms")
    plt.ylabel("Average run time")
    plt.title("Average run time of different algorithms\non graphs with " + str(num_vertices) + " vertices")
    plt.show()

    vc_len_fig = plt.figure(figsize=(15, 10))
    plt.bar(algo_names, vc_len_list_avg)
    plt.xlabel("Algorithms")
    plt.ylabel("Average Vertex Cover size")
    plt.title("Average Vertex Cover size of different algorithms\non graphs with " + str(num_vertices) + " vertices")
    plt.show()

    is_vc_fig = plt.figure(figsize=(15, 10))
    plt.bar(algo_names, is_vc_list_avg)
    plt.xlabel("Algorithms")
    plt.ylabel("Average success rate")
    plt.title("Average success rate of different algorithms\non graphs with " + str(num_vertices) + " vertices")
    plt.show()

def create_random_graph(num_of_vertices):
    """
    This function creates a random graph
    :param num_of_vertices: The amount of vertices in the graph
    :return: a Graph object
    """
    vertex_list = []
    for i in range(num_of_vertices):
        vertex_list.append(i)
    all_possible_edges = [frozenset({i, j}) for i in range(len(vertex_list) - 1) for j in range(i + 1, len(vertex_list))]
    edges_list = random.sample(all_possible_edges, random.randint(1, len(all_possible_edges)))
    neighbors = dict()
    for vertex in vertex_list:
        vertex_neighbors = set()
        for edge in edges_list:
            if vertex in edge:
                vertex_neighbors |= edge.difference({vertex})
        neighbors[vertex] = vertex_neighbors
    return Graph(edges_list, vertex_list, neighbors)

def build_graph_from_file(file_name):
    file = open(file_name, "r")
    first_line = file.readline().split()
    num_of_vertices = int(first_line[2])
    edges = []
    for line in file:
        line = line.split()
        if line[0] == "e":
            edges.append((int(line[1]) - 1, int(line[2]) - 1))
    graph = Graph()
    graph.create_graph(num_of_vertices, edges)
    print("finish graph build")
    file.close()
    return graph


def run_algorithm_on_graph(params, graph_name: str, num_iter: int, algorithm_type: Algorithm,
                           algorithm, initial_state: InitialState, run_hill_climbing=False):
    total_time = 0
    # total_hc_time = 0
    min_vc = math.inf
    total_vc_size = 0
    # res = [graph_name, algorithm.__name__, num_iter, initial_state.name, None, None, None, 0, "", "", 0, ""]
    res = [algorithm.__name__]
    func = algorithm
    rel_params = params
    if algorithm_type == Algorithm.GA:
        ga = algorithm(params[0])
        func = ga.perform_ga
        rel_params = params[1:]
        # res[4] = params[1]
        # res[5] = params[2]
    # elif algorithm_type == Algorithm.ITERATIVE:
        # res[4] = params[1]
    # elif algorithm_type == Algorithm.SA:
    #     temp_func = params[2]
        # bla = inspect.getsourcelines(temp_func)[0][0]
        # res[6] = bla[bla.find(":") + 1:]
    # elif algorithm_type == Algorithm.LBS:
        # res[5] = params[1]
    for j in range(num_iter):
        vc, run_time = get_time_and_state(func, rel_params)
        total_time += run_time
        # res[8] += str(len(vc)) + "|"
        # res[9] += str(is_vc(params[0], vc)) + "|"
        if run_hill_climbing:
            vc, extra_time = get_time_and_state(greedy_hill_climbing, [params[0], vc])
            total_time += extra_time
        if len(vc) < min_vc:
            min_vc = len(vc)
        total_vc_size += len(vc)
            # total_hc_time += extra_time
    #         res[11] += str(len(new_vc)) + "|"
    # res[7] = total_time / num_iter
    # res[10] = res[7] + total_hc_time / num_iter
    res.append(total_vc_size / num_iter)
    res.append(min_vc)
    res.append(total_time / num_iter)
    return res


def get_time_and_state(func, params):
    start_time = time.time()
    vc = func(*params)
    end_time = time.time()
    return vc, end_time - start_time


def xl1(graph_name: str, graph):
    # column_names = ["graph_name", "algorithm_type", "num_iter", "initial_state", "num_inner_iters/generations",
    #                 "population_size/num_children", "schedule", "avg_time", "vc_sizes", "is_vc", "avg_time_with_hc", "vc_sizes_with_hc"]
    column_names = ["Algorithm Name", "Avg VC Size", "Min VC Size", "Avg Run Time (sec)"]
    workbook = xl.Workbook(graph_name + ".xlsx")
    worksheet = workbook.add_worksheet()
    for i in range(len(column_names)):
        worksheet.write(0, i, column_names[i])
    results = []
    results.append(run_algorithm_on_graph([graph], graph_name, 10, Algorithm.TWO_APPROX, two_approximate_vertex_cover, InitialState.EMPTY))
    results.append(run_algorithm_on_graph([graph, []], graph_name, 10, Algorithm.HC, greedy_hill_climbing, InitialState.EMPTY))
    results.append(run_algorithm_on_graph([graph, []], graph_name, 10, Algorithm.HC, stochastic_hill_climbing, InitialState.EMPTY))
    # results.append(run_algorithm_on_graph([graph, []], graph_name, 2, Algorithm.HC, first_choice_hill_climbing, InitialState.EMPTY))
    results.append(run_algorithm_on_graph([graph, 100], graph_name, 10, Algorithm.ITERATIVE, random_restart_hill_climbing, InitialState.RANDOM))
    f = lambda x: 1 - 0.00001 * x
    results.append(run_algorithm_on_graph([graph, [], f, cost4], graph_name, 10, Algorithm.SA, simulated_annealing, InitialState.EMPTY, True))
    results.append(run_algorithm_on_graph([graph, 25], graph_name, 10, Algorithm.LBS, local_beam_search, InitialState.RANDOM))
    results.append(run_algorithm_on_graph([graph, 100], graph_name, 10, Algorithm.ITERATIVE, ghc_weighted_edges, InitialState.EMPTY))
    results.append(run_algorithm_on_graph([graph, []], graph_name, 10, Algorithm.HC, ghc_weighted_vertices, InitialState.EMPTY))
    # results.append(run_algorithm_on_graph([graph, 10], graph_name, 2, Algorithm.ITERATIVE, random_restart_whc_special2, InitialState.EMPTY))
    results.append(run_algorithm_on_graph([graph, 10000, round((0.315 * (graph.get_num_vertices() ** 0.6) + 0.72))], graph_name, 10, Algorithm.GA, RegularVC_GA, InitialState.RANDOM, True))
    results.append(run_algorithm_on_graph([graph, 10000, round((0.315 * (graph.get_num_vertices() ** 0.6) + 0.72))], graph_name, 10, Algorithm.GA, RegularVC_GA2, InitialState.RANDOM, True))
    results.append(run_algorithm_on_graph([graph, 10000, round((0.315 * (graph.get_num_vertices() ** 0.6) + 0.72))], graph_name, 10, Algorithm.GA, VCPunish_GA, InitialState.RANDOM, True))
    # results.append(run_algorithm_on_graph([graph, 10000, math.ceil((graph.get_num_vertices() ** 0.6) / 3)], graph_name, 2, Algorithm.GA, VC_NEW_MUT, InitialState.RANDOM, True))
    row = 1
    for result in results:
        for i in range(len(result)):
            worksheet.write(row, i, result[i])
        row += 1
    workbook.close()

if __name__ == '__main__':
    g = build_graph_from_file(".\\graph_files\\brock200_2.mis")
    xl1("brock200_2.mis", g)

#     a = glob.glob(".\graph_files\*.mis")
#     print(run_algorithm_on_files(a, 2, two_approximate_vertex_cover))