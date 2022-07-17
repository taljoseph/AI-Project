# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from Graph import *
import itertools
import random


def create_random_graph(num_of_vertices):
    """
    This function creates a random graph
    :param num_of_vertices: The amount of vertices in the graph
    :return: a Graph object
    """
    vertex_list = []
    for i in range(num_of_vertices):
        vertex_list.append(i)
    all_possible_edges = list(itertools.combinations(vertex_list, 2))
    edges_list = random.sample(all_possible_edges, random.randint(1, len(all_possible_edges)))
    neighbors = dict()
    vertex_edges_dic = dict()
    for vertex in vertex_list:
        vertex_neighbors = []
        vertex_edges = []
        for e in edges_list:
            if vertex == e[0]:
                vertex_neighbors.append(e[1])
                vertex_edges.append(e)
            elif vertex == e[1]:
                vertex_neighbors.append(e[0])
                vertex_edges.append(e)
        neighbors[vertex] = vertex_neighbors
        vertex_edges_dic[vertex] = vertex_edges
    return Graph(edges_list, len(edges_list), vertex_list, num_of_vertices, neighbors, vertex_edges_dic)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    new_graph = create_random_graph(5)
    vertex_cover = new_graph.two_approximate_vertex_cover()
    print(vertex_cover)
    new_graph.draw_vertex_cover(vertex_cover)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
