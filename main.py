# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import copy

from Graph import *
from algorithms import *
import itertools
import random
from vc_problem import *
import time


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
    # graph = Graph([frozenset({0, 1}), frozenset({0, 2}), frozenset({0, 3})], [0, 1, 2, 3], {0: {2, 3, 1}, 2: {0}, 3: {0}, 1: {0}})
    # graph = Graph([frozenset({0, 4}), frozenset({1, 2}), frozenset({1, 4}), frozenset({2, 3})], [0, 1, 2, 3, 4], {0: {4}, 1: {2, 4}, 2: {1, 3}, 3: {2}, 4: {0, 1}})
    graph = Graph()
    graph.create_p_random_graph(200, 0.005)
    # print(graph)
    starta = time.time()
    a = greedy_hill_climbing(graph, [])
    print(time.time() - starta)
    startb = time.time()
    b = two_approximate_vertex_cover(graph)
    print(time.time() - startb)
    print(a)
    print(len(a), is_goal_state(graph, a))
    print(b)
    print(len(b), is_goal_state(graph, b))
    graph.draw_vertex_cover(a)
    graph.draw_vertex_cover(b)
    # problem = VC_Problem(graph, [1, 2])
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
