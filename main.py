from algorithms import *
import sys
import pandas as pd
import time


ALG_NAME = "Algorithm name"
VC_SIZES = "VC Sizes"
AVG_TIME = "Average time"
MIN_VC = "Min VC"


def run_for_user(algorithm: str, graph_type: str, iters: str, res_path: str):
    """
    Runs an algorithm on a certain graph for a number of iterations
    :param algorithm: All/one of the amazing algorithms we created
    :param graph_type: prandome_numVertices_p/random_numVertices_Numedged/path
    :param iters: number of iterations.
    :param res_path: Result csv file path
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
    results_dict = {ALG_NAME: [], VC_SIZES: [], AVG_TIME: [], MIN_VC: []}
    if algorithm == "all":
        for alg in algo_dict.keys():
            run_for_user_helper(algo_dict[alg], graph, iterations, results_dict)
    else:
        run_for_user_helper(algo_dict[algorithm], graph, iterations, results_dict)
    df = pd.DataFrame(results_dict)
    df.to_csv(res_path, index=False)


def run_for_user_helper(algorithm, graph: Graph, iters: int, results_dict):
    """
    Checks which algorithm to run and runs it.
    :param algorithm: algorithm function
    :param graph: graph
    :param iters: number of iterations to run the algorithm
    :param results_dict: dictionary to write the results to
    """

    if algorithm == two_approximate_vertex_cover:
        run_iter_times(algorithm, [graph], iters, algorithm.__name__, results_dict)

    elif algorithm == greedy_hill_climbing or algorithm == ghc_weighted_vertices or algorithm == stochastic_hill_climbing:
        run_iter_times(algorithm, [graph, []], iters, algorithm.__name__, results_dict)

    elif algorithm == random_restart_hill_climbing:
        run_iter_times(algorithm, [graph, 100, greedy_hill_climbing], iters, algorithm.__name__, results_dict)

    elif algorithm == ghc_weighted_edges:
        run_iter_times(algorithm, [graph, 100], iters, algorithm.__name__, results_dict)

    elif algorithm == simulated_annealing:
        run_iter_times(algorithm, [graph, [], lambda x: 1 - 0.00001 * x, cost4], iters, algorithm.__name__, results_dict)

    elif algorithm == local_beam_search:
        run_iter_times(algorithm, [graph, 25], iters, algorithm.__name__, results_dict)

    elif algorithm == RegularVC_GA or algorithm == RegularVC_GA2 or algorithm == VCPunish_GA:
        ga_graph = algorithm(graph)
        run_iter_times(None, [10000, round((0.315 * (graph.get_num_vertices() ** 0.6) + 0.72))], iters,
                       algorithm.__name__, results_dict, ga_graph)

    else:
        print("Invalid algorithm provided")
        return


def run_iter_times(algo, params, iters, algo_name, results_dict, ga_graph=None):
    """
    Runs the algorithm on the params for iters iterations
    :param algo: algorithm function
    :param params: parameters for the algorithm function
    :param iters: number of iterations to run
    :param algo_name: algorithm name
    :param results_dict: dictionary to write results into
    :param ga_graph: In case of GA
    """
    vcs = []
    min_vc_size = float("inf")
    min_vc = []
    start = time.time()
    for i in range(iters):
        if ga_graph is not None:
            vc = ga_graph.perform_ga(*params)
        else:
            vc = algo(*params)
        if len(vc) < min_vc_size:
            min_vc_size = len(vc)
            min_vc = vc
        vcs.append(len(vc))
    avg_run_time = (time.time() - start) / iters
    results_dict[ALG_NAME].append(algo_name)
    results_dict[VC_SIZES].append(vcs)
    results_dict[AVG_TIME].append(avg_run_time)
    results_dict[MIN_VC].append(min_vc)


if __name__ == '__main__':
    _, algorithm_name, graph_type_path, num_iters, res_path = sys.argv
    run_for_user(algorithm_name, graph_type_path, num_iters, res_path)
