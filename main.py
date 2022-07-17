# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import copy

from Graph import *
from algorithms import *
import itertools
import random
from vc_problem import *


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
    graph = Graph([frozenset({1, 2}), frozenset({1, 3}), frozenset({1, 4})], [1, 2, 3, 4])
    problem = VC_Problem(graph, [1, 2])
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
