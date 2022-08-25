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

class Algorithm(Enum):
    TWO_APPROX = 1  # graph
    HC = 2   # Hill Climbing: graph, initial state
    ITERATIVE = 3  # Iterative Hill Climbing: graph, num iterations
    SA = 4  # Simulated Annealing: graph, initial state, schedule function
    LBS = 5  # Local Beam Search: graph, num children
    GA = 6  # Genetic Algorithms: num gens, population size


def build_graph_from_file(file_name):
    """
    This function builds a graph from the file
    @param file_name: the file name to create a graph from
    @return: a graph
    """
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


def run_algorithm_on_graph(params, num_iter: int, algorithm_type: Algorithm,
                           algorithm, run_hill_climbing=False):
    """
    This function run a certain algorithm and it's parameters on a certain graph = params[0]
    @param params: the parameters of the algorithm, graph = params[0]
    @param num_iter: the number of iterations to run the algorithm on the graph
    @param algorithm_type: the type of the algorithm, an enum
    @param algorithm: the algorithm to run
    @param run_hill_climbing: True if needed to run hill climbing, else false
    @return: results = [algorithm name, avg vc size, min vc size, avg run time (sec)]
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
    This function runs the func with it's params and counts the time it takes
    @param func: the function to run
    @param params: the parameters of the func
    @return: time it took to run the function
    """
    start_time = time.time()
    vc = func(*params)
    end_time = time.time()
    return vc, end_time - start_time


def xl1(graph_name: str, graph):
    """
    Creates a .xlsx document which holds the results of running the different algorithms on a certain graph
    @param graph_name: the name of the graph
    @param graph: the graph to run the algorithms on
    @return:
    """
    column_names = ["Algorithm Name", "Avg VC Size", "Min VC Size", "Avg Run Time (sec)"]
    workbook = xl.Workbook(graph_name + ".xlsx")
    worksheet = workbook.add_worksheet()
    for i in range(len(column_names)):
        worksheet.write(0, i, column_names[i])
    results = []
    results.append(run_algorithm_on_graph([graph], 10, Algorithm.TWO_APPROX, two_approximate_vertex_cover))
    results.append(run_algorithm_on_graph([graph, []], 10, Algorithm.HC, greedy_hill_climbing))
    results.append(run_algorithm_on_graph([graph, []], 10, Algorithm.HC, stochastic_hill_climbing))
    results.append(run_algorithm_on_graph([graph, 100], 10, Algorithm.ITERATIVE, random_restart_hill_climbing))
    f = lambda x: 1 - 0.00001 * x
    results.append(run_algorithm_on_graph([graph, [], f, cost4], 10, Algorithm.SA, simulated_annealing, True))
    results.append(run_algorithm_on_graph([graph, 25], 10, Algorithm.LBS, local_beam_search))
    results.append(run_algorithm_on_graph([graph, 100], 10, Algorithm.ITERATIVE, ghc_weighted_edges))
    results.append(run_algorithm_on_graph([graph, []], 10, Algorithm.HC, ghc_weighted_vertices))
    results.append(run_algorithm_on_graph([graph, 10000, round((0.315 * (graph.get_num_vertices() ** 0.6) + 0.72))], 10,
                                          Algorithm.GA, RegularVC_GA, True))
    results.append(run_algorithm_on_graph([graph, 10000, round((0.315 * (graph.get_num_vertices() ** 0.6) + 0.72))], 10,
                                          Algorithm.GA, RegularVC_GA2, True))
    results.append(run_algorithm_on_graph([graph, 10000, round((0.315 * (graph.get_num_vertices() ** 0.6) + 0.72))], 10,
                                          Algorithm.GA, VCPunish_GA, True))
    row = 1
    for result in results:
        for i in range(len(result)):
            worksheet.write(row, i, result[i])
        row += 1
    workbook.close()

if __name__ == '__main__':
    # g = build_graph_from_file(".\\graph_files\\keller4.mis")
    # xl1("keller4.mis", g)
    # g = build_graph_from_file(".\\graph_files\\MANN_a27.mis")
    # xl1("MANN_a27.mis", g)
    # g = build_graph_from_file(".\\graph_files\\MANN_a45.mis")
    # xl1("MANN_a45.mis", g)
    # g = build_graph_from_file(".\\graph_files\\MANN_a81.mis")
    # xl1("MANN_a81.mis", g)
    # g = build_graph_from_file(".\\graph_files\\C500.9.mis")
    # xl1("C500.9.mis", g)
    g = build_graph_from_file(".\\graph_files\\gen200_p0.9_44.mis")
    xl1("gen200_p0.9_44.mis", g)








#     a = glob.glob(".\graph_files\*.mis")
#     print(run_algorithm_on_files(a, 2, two_approximate_vertex_cover))