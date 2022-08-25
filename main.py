# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import copy
import csv

from graph_numpy import *
from Graph import *
from algorithms import *
import itertools
import random
from vc_problem import *
import time
from genetic_algorithms import *
import networkx as nx
from utils import *

def run_for_user(algorithm: str, graph_type: str, iters: str):
    if "prandom" in graph_type:
        num_vertices, edges_ratio = graph_type.split("_")[1:]
        graph = Graph()
        graph.create_p_random_graph(int(num_vertices), float(edges_ratio))
    elif "random" in graph_type:
        num_vertices, num_edges = graph_type.split("_")[1:]
        graph = Graph()
        graph.create_nx_graph(int(num_vertices), int(num_edges))
    else:
        graph = build_graph_from_file(".\\graph_files\\" + graph_type)
    algo_dict = {"two_approximate_vertex_cover": two_approximate_vertex_cover, "greedy_hill_climbing": greedy_hill_climbing,
                "random_restart_hill_climbing": random_restart_hill_climbing, "ghc_weighted_edges": ghc_weighted_edges,
                "ghc_weighted_vertices": ghc_weighted_vertices, "simulated_annealing": simulated_annealing,
                "local_beam_search": local_beam_search, "stochastic_hill_climbing": stochastic_hill_climbing,
                "RegularVC_GA": RegularVC_GA, "RegularVC_GA2": RegularVC_GA2, "VCPunish_GA": VCPunish_GA}
    iterations = int(iters)
    f = open(".\\RESULTS.csv", 'w')
    writer = csv.writer(f)
    writer.writerow(["Algo Name", "VC size"])
    if algorithm == "all":
        run_for_user_helper(algorithm, graph, iterations, f)
    else:
        run_for_user_helper(algo_dict[algorithm], graph, iterations, f)
    f.close()


def run_for_user_helper(algorithm, graph: Graph, iters: int, file):

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

def run_iter_times(algo, params, iters, algo_name, file, ga_graph=None ):
    writer = csv.writer(file)
    results = ""
    for i in range(iters):
        if ga_graph is not None:
            vc = ga_graph.perform_ga(*params)
        else:
            vc = algo(*params)
        results += str(len(vc)) + " ,"

    writer.writerow([algo_name, results])


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



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_for_user("all", "C125.9.mis", "2")
    # new_graph = create_random_graph(10)
    # new_graph = Graph()
    # new_graph.create_p_random_graph(100, 0.05)
    # vertex_cover = two_approximate_vertex_cover(new_graph)
    # print(vertex_cover)
    # new_graph.draw_vertex_cover(vertex_cover)
    # graph1 = Graph([frozenset({0, 1}), frozenset({0, 2}), frozenset({0, 3})], [0, 1, 2, 3], {0: {2, 3, 1}, 2: {0}, 3: {0}, 1: {0}})
    # graph1 = Graph([frozenset({0, 4}), frozenset({1, 2}), frozenset({1, 4}), frozenset({2, 3})], [0, 1, 2, 3, 4], {0: {4}, 1: {2, 4}, 2: {1, 3}, 3: {2}, 4: {0, 1}})
    # graph1 = Graph()
    # graph1.create_p_random_graph(200, 0.005)
    # # print(graph1)
    # starta = time.time()
    # a = greedy_hill_climbing(graph1, [])
    # print(time.time() - starta)
    # startb = time.time()
    # b = two_approximate_vertex_cover(graph1)
    # print(time.time() - startb)
    # print(a)
    # print(len(a), is_goal_state(graph1, a))
    # print(b)
    # print(len(b), is_goal_state(graph1, b))
    # graph1.draw_vertex_cover(a)
    # graph1.draw_vertex_cover(b)
    # problem = VC_Problem(graph1, [1, 2])

    # num_vertices = 500
    # graph1 = Graph()
    # graph1.create_p_random_graph(num_vertices, 0.0035)
    # graph1.create_nx_graph(200, math.ceil(19900 * 0.20))



    # start1 = time.time()
    # vc1 = two_approximate_vertex_cover(graph1)
    # time1 = time.time() - start1
    # print("Two Approx:\nNum vertices: {}\nis_cover: {}\ntime(sec): {}\n".format(len(vc1), is_vc(graph1, vc1), time1))
    #
    # start2 = time.time()
    # vc2 = greedy_hill_climbing(graph1, [])
    # time2 = time.time() - start1
    # print("Hill Climbing:\nNum vertices: {}\nis_cover: {}\ntime(sec): {}\n".format(len(vc2), is_vc(graph1, vc2), time2))
    # #
    # # vc_ga = RegularVC_GA(graph1)
    # # start3 = time.time()
    # # vc3 = vc_ga.perform_ga(10000, math.ceil((num_vertices ** 0.6) / 3))
    # # time3 = time.time() - start3
    # # print("Genetic Alg:\nNum vertices: {}\nis_cover: {}\ntime(sec): {}\n".format(len(vc3), is_vc(graph1, vc3), time3))
    # #
    # # start4 = time.time()
    # # vc4 = ghc_weighted(graph1, [], 100)
    # # time4 = time.time() - start4
    # # print("Weighted Hill Climbing:\nNum vertices: {}\nis_cover: {}\ntime(sec): {}\n".format(len(vc4), is_vc(graph1, vc4), time4))
    #
    # # # start5 = time.time()
    # # # vc5 = random_restart_hill_climbing(graph1, 100)
    # # # time5 = time.time() - start5
    # # # print("Random Restart Hill Climbing:\nNum vertices: {}\nis_cover: {}\ntime(sec): {}\n".format(len(vc5), is_vc(graph1, vc5), time5))
    #
    # vc_ga_punish = RegularVC_GA2(graph1)
    # for i in range(1, 51):
    #     start6 = time.time()
    #     vc6 = vc_ga_punish.perform_ga(10000, p)
    #     time6 = time.time() - start6
    #     print("Genetic Alg punish:\nNum vertices: {}\nis_cover: {}\ntime(sec): {}\n".format(len(vc6), is_vc(graph1, vc6), time6))
    # #
    # start7 = time.time()
    # vc7 = greedy_hill_climbing(graph1, vc6)
    # time7 = time.time() - start7 + time6
    # print("Genetic Alg punish + hill:\nNum vertices: {}\nis_cover: {}\ntime(sec): {}\n".format(len(vc7), is_vc(graph1, vc7), time7))
    #
    #
    # start8 = time.time()
    # vc8 = ghc_weighted_special(graph1, [])
    # time8 = time.time() - start8
    # print("Weighted special 1:\nNum vertices: {}\nis_cover: {}\ntime(sec): {}\n".format(len(vc8), is_vc(graph1, vc8), time8))
    #
    # # start9 = time.time()
    # # vc9 = ghc_weighted_special2(graph1, [])
    # # time9 = time.time() - start9
    # # print("Weighted special 2:\nNum vertices: {}\nis_cover: {}\ntime(sec): {}\n".format(len(vc9), is_vc(graph1, vc9), time9))
    # #
    # # # start10 = time.time()
    # # # vc10 = greedy_hill_climbing(graph1, vc9)
    # # # time10 = time.time() - start10 + time9
    # # # print("weighted Special 2 + hill:\nNum vertices: {}\nis_cover: {}\ntime(sec): {}\n".format(len(vc10), is_vc(graph1, vc10), time10))
    # #
    # # start11 = time.time()
    # # vc11 = random_restart_whc_special2(graph1, 10)
    # # time11 = time.time() - start11
    # # print("random weighted Special 2:\nNum vertices: {}\nis_cover: {}\ntime(sec): {}\n".format(len(vc11), is_vc(graph1, vc11), time11))

    # vc_new = VC_NEW_MUT(graph1)
    # start12 = time.time()
    # vc12 = vc_new.perform_ga(10000, math.ceil((num_vertices ** 0.6) / 3))
    # time12 = time.time() - start12
    # print("Genetic Alg NEW:\nNum vertices: {}\nis_cover: {}\ntime(sec): {}\n".format(len(vc12), is_vc(graph1, vc12), time12))
    #
    # start13 = time.time()
    # vc13 = greedy_hill_climbing(graph1, vc12)
    # time13 = time.time() - start13 + time12
    # print("Genetic Alg NEW + hill:\nNum vertices: {}\nis_cover: {}\ntime(sec): {}\n".format(len(vc13), is_vc(graph1, vc13), time13))
    #


    # graph1.draw_vertex_cover(vc8)



    #
    # vc_ga_v2 = RegularVC_GA_v2(graph1)
    # vcc = vc_ga_v2.perform_ga(1000, 50)
    #

    import math
    #
    #
    # vertices = np.arange(9).tolist()
    # edge_list = [frozenset({0,1}), frozenset({0,3}), frozenset({1,2}), frozenset({1,3}), frozenset({2,3}),
    #              frozenset({3,4}), frozenset({4,5}), frozenset({4,7}), frozenset({5,6}), frozenset({5,7}),
    #              frozenset({5,8}), frozenset({6,7}), frozenset({7,8})]
    # neighbours = {0: {1, 3}, 1: {0, 2, 3}, 2: {1, 3}, 3: {0, 1, 2, 4}, 4: {3, 5, 7}, 5: {4, 6, 7, 8},
    #               6: {5, 7}, 7: {4, 5, 6, 8}, 8: {5, 7}}
    #
    # new_g = Graph(edge_list, vertices, neighbours)
    #
    # start1 = time.time()
    # vc1 = two_approximate_vertex_cover(new_g)
    # time1 = time.time() - start1
    # print("Two Approx:\nNum vertices: {}\nis_cover: {}\ntime(sec): {}\n".format(len(vc1), is_vc(new_g, vc1), time1))
    #
    # start2 = time.time()
    # vc2 = greedy_hill_climbing(new_g, [])
    # time2 = time.time() - start1
    # print("Hill Climbing:\nNum vertices: {}\nis_cover: {}\ntime(sec): {}\n".format(len(vc2), is_vc(new_g, vc2), time2))
    #
    # start4 = time.time()
    # vc4 = ghc_weighted(new_g, [], 100)
    # time4 = time.time() - start4
    # print("Weighted Hill Climbing:\nNum vertices: {}\nis_cover: {}\ntime(sec): {}\n".format(len(vc4), is_vc(new_g, vc4), time4))
    #
    #
    # start8 = time.time()
    # vc8 = ghc_weighted_special(new_g, [])
    # time8 = time.time() - start8
    # print("Weighted new:\nNum vertices: {}\nis_cover: {}\ntime(sec): {}\n".format(len(vc8), is_vc(new_g, vc8), time8))
    #
    #
    # new_g.draw_vertex_cover(vc8)

    # f = open(".\\graph_files\\C125.9.mis", 'r')
    # graph2 = build_graph_from_file(f)
    # a = two_approximate_vertex_cover(graph2)
    # graph2.draw_vertex_cover(a)
    # f.close()

    # # print(len(vc3))
    # # print(len(vc4))
    # graph1.draw_vertex_cover(vc3)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
#     vertices_num = [50, 80, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000]
#     edges_percentage = [0.1, 0.1, 0.08, 0.06, 0.03, 0.01, 0.007, 0.0035, 0.002, 0.001, 0.0008]
#     for j in range(10, len(vertices_num)):
#         graph1 = Graph()
#         graph1.create_p_random_graph(vertices_num[j], edges_percentage[j])
#         vc_ga_punish = RegularVC_GA2(graph1)
#         li = []
#         print(vertices_num[j])
#         start = time.time()
#         for i in range(1, 51):
#             print(i)
#             vc6 = greedy_hill_climbing(graph1, vc_ga_punish.perform_ga(10000, i))
#             li.append((i, len(vc6)))
#         print(li)
#         print(time.time() - start)
#     graph2 = Graph()
#     graph2.create_old_city_graph()
#
#
#     # graph2 = build_graph_from_file(".\\graph_files\\C250.9.mis")
#     start4 = time.time()
#     vc4 = multirun_whc_weighted_vertices(graph2, 1)
#     time4 = time.time() - start4
#     print("Weighted vertices new:\nNum vertices: {}\nis_cover: {}\ntime(sec): {}\n".format(len(vc4), is_vc(graph2, vc4), time4))

    # start5 = time.time()
    # vc5 = ghc_weighted_vertices_old(graph2, [])
    # time5 = time.time() - start5
    # print("Weighted vertices old:\nNum vertices: {}\nis_cover: {}\ntime(sec): {}\n".format(len(vc5), is_vc(graph2, vc5), time5))
    # bla = "p_random_10_0.004"
    # if "p_random" in bla:
    #     print("hi")
    #
    # print(bla.split("_")[2:])
    # graph = Graph()
    # graph.create_nx_graph(40, 30)
    # print(graph)