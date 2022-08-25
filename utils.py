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
import pandas as pd
import openpyxl


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


def create_random_graph_file(num_vertices, num_edges, path):
    nx_graph = nx.gnm_random_graph(num_vertices, num_edges)
    g = Graph()
    g.create_graph(num_vertices, nx_graph.edges)
    create_graph_file(g, path)


def create_p_random_graph_file(num_vertices, edges_ratio, path):
    g = Graph()
    g.create_p_random_graph(num_vertices, edges_ratio)
    create_graph_file(g, path)


def create_graph_file(graph, path):
    f = open(path, "w")
    f.write("p edge {} {}\n".format(graph.get_num_vertices(), graph.get_num_edges()))
    for edge in graph.get_edges():
        v, u = edge
        f.write("e {} {}\n".format(v + 1, u + 1))
    f.close()


def run_algorithm_on_graph(params, num_iter: int, algorithm_type: Algorithm,
                           algorithm, initial_state: InitialState = InitialState.EMPTY,
                           run_hill_climbing=False, num_vert_start=0):
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
            num_vert = random.randint(0, params[0].get_num_vertices())
            new_state = random.sample(params[0].get_vertices(), num_vert)
            rel_params[1] = new_state
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
    start_time = time.time()
    vc = func(*params)
    end_time = time.time()
    return vc, end_time - start_time


def xl1(graph_name: str, graph):
    column_names = ["Algorithm Name", "Avg VC Size", "Min VC Size", "Avg Run Time (sec)"]
    workbook = xl.Workbook(graph_name + ".xlsx")
    worksheet = workbook.add_worksheet()
    init_state = InitialState.TWO_APPROX_COVER
    for i in range(len(column_names)):
        worksheet.write(0, i, column_names[i])
    results = []
    two_approx_vc = run_algorithm_on_graph([graph], 10, Algorithm.TWO_APPROX, two_approximate_vertex_cover)
    start_vertices = two_approx_vc[2] // 2
    results.append(run_algorithm_on_graph([graph, []], 10, Algorithm.HC, greedy_hill_climbing,
                                          init_state, run_hill_climbing=False, num_vert_start=start_vertices))
    results.append(run_algorithm_on_graph([graph, []], 10, Algorithm.HC, stochastic_hill_climbing,
                                          init_state, run_hill_climbing=False, num_vert_start=start_vertices))
    # # results.append(run_algorithm_on_graph([graph, []], 2, Algorithm.HC, first_choice_hill_climbing, InitialState.EMPTY))
    # results.append(run_algorithm_on_graph([graph, 100], 10, Algorithm.ITERATIVE, random_restart_hill_climbing, InitialState.RANDOM))
    results.append(run_algorithm_on_graph([graph, [], lambda x: 1 - 0.00001 * x, cost4], 10, Algorithm.SA,
                                          simulated_annealing, init_state, run_hill_climbing=True, num_vert_start=start_vertices))
    # results.append(run_algorithm_on_graph([graph, 25], 10, Algorithm.LBS, local_beam_search, InitialState.RANDOM))
    # results.append(run_algorithm_on_graph([graph, 100], 1, Algorithm.ITERATIVE, ghc_weighted_edges, InitialState.EMPTY))
    results.append(run_algorithm_on_graph([graph, []], 10, Algorithm.HC, ghc_weighted_vertices,
                                          init_state, run_hill_climbing=False, num_vert_start=start_vertices))
    # # results.append(run_algorithm_on_graph([graph, 10], 2, Algorithm.ITERATIVE, random_restart_whc_special2, InitialState.EMPTY))
    # results.append(run_algorithm_on_graph([graph, 10000, round((0.315 * (graph.get_num_vertices() ** 0.6) + 0.72))], 10, Algorithm.GA, RegularVC_GA, InitialState.RANDOM, True))
    # results.append(run_algorithm_on_graph([graph, 10000, round((0.315 * (graph.get_num_vertices() ** 0.6) + 0.72))], 10, Algorithm.GA, RegularVC_GA2, InitialState.RANDOM, True))
    # results.append(run_algorithm_on_graph([graph, 10000, round((0.315 * (graph.get_num_vertices() ** 0.6) + 0.72))], 10, Algorithm.GA, VCPunish_GA, InitialState.RANDOM, True))
    # # results.append(run_algorithm_on_graph([graph, 10000, math.ceil((graph.get_num_vertices() ** 0.6) / 3)], 2, Algorithm.GA, VC_NEW_MUT, InitialState.RANDOM, True))
    row = 1
    for result in results:
        for i in range(len(result)):
            worksheet.write(row, i, result[i])
        row += 1
    workbook.close()


def best_algo_graph(path):
    files = glob.glob(path)
    sum_avg_values = np.zeros(4)
    for file in files:
        df = pd.read_excel(file)
        a = df["Avg VC Size"]
        a = a.to_numpy()
        sum_avg_values += a
    return sum_avg_values


if __name__ == '__main__':
    # graphs = ["gen400_p0.9_75.mis",
    #           "gen400_p0.9_65.mis", "gen400_p0.9_55.mis", "gen200_p0.9_55.mis"]
    # "p_hat300-2.mis", "p_hat300-3.mis", "MANN_a81.mis", "MANN_a45.mis",
    #               "MANN_a27.mis", "keller4.mis", "hamming8-4.mis",
    #
    # i = 1000
    # for graph in graphs:
    #     g = build_graph_from_file(".\\graph_files\\" + graph)
    #     xl1("{}".format(i), g)
    #     i+=1
    # g = Graph()
    # g.create_old_city_graph()
    # f = build_graph_from_file(".\\graph_files\\p_hat300-3.mis")
    # xl1("sdkjhf", f)

    # all_g = ["brock200_2.mis", "brock200_4.mis", "C125.9.mis", "C250.9.mis", "C500.9.mis", "gen200_p0.9_44.mis",
    #          "gen200_p0.9_55.mis", "gen400_p0.9_55.mis", "gen400_p0.9_65.mis", "gen400_p0.9_75.mis",
    #          "hamming8-4.mis", "keller4.mis", "MANN_a27.mis", "MANN_a45.mis", "MANN_a81.mis", "p_hat300-3.mis",
    #          "p_hat300-2.mis"]
    # for g in all_g:
    #     graph = build_graph_from_file(".\\graph_files\\" + g)
    #     xl1(g, graph)

    # g1 = Graph()
    # g1.create_p_random_graph(500, 0.006)
    # g2 = Graph()
    #
    #
    # f = open("C:\\Users\\talyo\\OneDrive\\Desktop\\AI-Project\\p-random_graph.txt", "w")
    # f.write("p edge {} {}\n".format(g1.get_num_vertices(), g1.get_num_edges()))
    # for edge in g1.get_edges():
    #     v, u = edge
    #     f.write("e {} {}\n".format(v, u))
    # f.close()

    # create_p_random_graph_file(500, 0.006, "C:\\Users\\talyo\\OneDrive\\Desktop\\AI-Project\\p_random_graph.txt")
    # create_random_graph_file(500, 5000, "C:\\Users\\talyo\\OneDrive\\Desktop\\AI-Project\\random_graph.txt")

    # g = build_graph_from_file(".\\p_random_graph.txt")
    # f = build_graph_from_file(".\\random_graph.txt")
    # xl1("p-random_graph", g)
    # xl1("random_graph", f)
    #
    # h = Graph()
    # h.create_old_city_graph()
    # xl1("Old_City_Jerusalem", h)

    print(best_algo_graph(".\\graphs - using empty state\*.xlsx"))
    print(best_algo_graph(".\\graphs - using full state\*.xlsx"))
    print(best_algo_graph(".\\graphs - using half two-approx\*.xlsx"))
    print(best_algo_graph(".\\graphs - using random state\*.xlsx"))
    # g = build_graph_from_file(".\\graph_files\\MANN_a45.mis")
    # for i in range(5):
    #     xl1("1000" + str(i), g)



#     a = glob.glob(".\graph_files\*.mis")
#     print(run_algorithm_on_files(a, 2, two_approximate_vertex_cover))