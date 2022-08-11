# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import copy

from graph_numpy import *
from Graph import *
from algorithms import *
import itertools
import random
from vc_problem import *
import time
from genetic_algorithms import *
import networkx as nx
from utils import  *

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

    num_vertices = 500
    graph1 = Graph()
    graph1.create_p_random_graph(num_vertices, 0.01)
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
    # vc_ga_punish = VCPunish_GA(graph1)
    # start6 = time.time()
    # vc6 = vc_ga_punish.perform_ga(10000, math.ceil((num_vertices ** 0.6) / 3))
    # time6 = time.time() - start6
    # print("Genetic Alg punish:\nNum vertices: {}\nis_cover: {}\ntime(sec): {}\n".format(len(vc6), is_vc(graph1, vc6), time6))
    #
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

    f = open(".\\graph_files\\C125.9.mis", 'r')
    graph2 = build_graph_from_file(f)
    a = two_approximate_vertex_cover(graph2)
    graph2.draw_vertex_cover(a)
    f.close()

    # # print(len(vc3))
    # # print(len(vc4))
    # graph1.draw_vertex_cover(vc3)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
