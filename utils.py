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

def build_graph_from_file(file):
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
    return graph

def run_algorithm_on_files(files_names, num_of_iter, algorithm = None, params = None, genetic = False):
    output = []
    for i in range(len(files_names)):
        file = open(files_names[i], "r")
        graph = build_graph_from_file(file)
        total_time = 0
        total_vc_size = 0
        for j in range(num_of_iter):
            vc = []
            start_time = 0
            end_time = 0
            if genetic:
                ga = RegularVC_GA(graph)
                start_time = time.time()
                vc = ga.perform_ga(params[0], params[1])
                end_time = time.time()
            else:
                start_time = time.time()
                if params:
                    vc = algorithm(graph, *params)
                else:
                    vc = algorithm(graph)
                end_time = time.time()
            total_time += (end_time - start_time)
            total_vc_size += len(vc)
        res = [files_names[i], num_of_iter, total_time/num_of_iter, total_vc_size/num_of_iter]
        print(res)
        output.append(res)
    return output


# running example
# if __name__ == '__main__':

#     a = glob.glob(".\graph_files\*.mis")
#     print(run_algorithm_on_files(a, 2, two_approximate_vertex_cover))