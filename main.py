import csv
from utils import *
import sys


def run_for_user(algorithm: str, graph_type: str, iters: str):
    """
    Runs an algorithm on a certain graph for a number of iterations
    :param algorithm: All/one of the amazing algorithms we created
    :param graph_type: prandome_numVertices_p/random_numVertices_Numedged/path
    :param iters: number of iterations.
    """
    if "prandom" in graph_type:
        num_vertices, edges_ratio = graph_type.split("_")[1:]
        graph = Graph()
        graph.create_p_random_graph(int(num_vertices), float(edges_ratio))
    elif "random" in graph_type:
        num_vertices, num_edges = graph_type.split("_")[1:]
        graph = Graph()
        graph.create_nx_graph(int(num_vertices), int(num_edges))
    else:
        graph = build_graph_from_file(graph_type)
    algo_dict = {"two_approximate_vertex_cover": two_approximate_vertex_cover, "greedy_hill_climbing": greedy_hill_climbing,
                "random_restart_hill_climbing": random_restart_hill_climbing, "ghc_weighted_edges": ghc_weighted_edges,
                "ghc_weighted_vertices": ghc_weighted_vertices, "simulated_annealing": simulated_annealing,
                "local_beam_search": local_beam_search, "stochastic_hill_climbing": stochastic_hill_climbing,
                "RegularVC_GA": RegularVC_GA, "RegularVC_GA2": RegularVC_GA2, "VCPunish_GA": VCPunish_GA}
    iterations = int(iters)
    f = open(".\\RESULTS.csv", 'w')
    writer = csv.writer(f)
    writer.writerow(["Algo Name", "VC size", "Min VC"])
    if algorithm == "all":
        run_for_user_helper(algorithm, graph, iterations, f)
    else:
        run_for_user_helper(algo_dict[algorithm], graph, iterations, f)
    f.close()


def run_for_user_helper(algorithm, graph: Graph, iters: int, file):
    """
    Checks which algorithm to run and runs it.
    :param algorithm: algorithm function
    :param graph: graph
    :param iters: number of iterations to run the algorithm
    :param file: file to write the results into
    """

    if algorithm == two_approximate_vertex_cover:
        run_iter_times(algorithm, [graph], iters, algorithm.__name__, file)

    elif algorithm == greedy_hill_climbing or algorithm == ghc_weighted_vertices or algorithm == stochastic_hill_climbing:
        run_iter_times(algorithm, [graph, []], iters, algorithm.__name__, file)

    elif algorithm == random_restart_hill_climbing:
        run_iter_times(algorithm, [graph, 100, greedy_hill_climbing], iters, algorithm.__name__, file)

    elif algorithm == ghc_weighted_edges:
        run_iter_times(algorithm, [graph, 100], iters, algorithm.__name__, file)

    elif algorithm == simulated_annealing:
        run_iter_times(algorithm, [graph, [], lambda x: 1 - 0.00001 * x, cost4], iters, algorithm.__name__, file)

    elif algorithm == local_beam_search:
        run_iter_times(algorithm, [graph, 25], iters, algorithm.__name__, file)

    elif algorithm == RegularVC_GA or algorithm == RegularVC_GA2 or algorithm == VCPunish_GA:
        ga_graph = algorithm(graph)
        run_iter_times(None, [10000, round((0.315 * (graph.get_num_vertices() ** 0.6) + 0.72))], iters,
                       algorithm.__name__, file, ga_graph)

    elif algorithm == "all":
        all_algo = [two_approximate_vertex_cover, greedy_hill_climbing, random_restart_hill_climbing,
                    ghc_weighted_edges, ghc_weighted_vertices, simulated_annealing, local_beam_search,
                    stochastic_hill_climbing, RegularVC_GA, RegularVC_GA2, VCPunish_GA]
        for alg in all_algo:
            run_for_user_helper(alg, graph, iters, file)
    else:
        print("Invalid algorithm provided")
        return


def run_iter_times(algo, params, iters, algo_name, file, ga_graph=None):
    """
    Runs the algorithm on the params for iters iterations
    :param algo: algorithm function
    :param params: parameters for the algorithm function
    :param iters: number of iterations to run
    :param algo_name: algorithm name
    :param file: file to write results into
    :param ga_graph: In case of GA
    """
    writer = csv.writer(file)
    results = ""
    min_vc_size = float("inf")
    min_vc = []
    for i in range(iters):
        if ga_graph is not None:
            vc = ga_graph.perform_ga(*params)
        else:
            vc = algo(*params)
        if len(vc) < min_vc_size:
            min_vc_size = len(vc)
            min_vc = vc
        results += str(len(vc)) + " ,"

    writer.writerow([algo_name, results, min_vc])


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


if __name__ == '__main__':
    _, first, second, third = sys.argv
    run_for_user(first, second, third)
