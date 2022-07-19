import networkx as nx
import random
import matplotlib.pyplot as plt
from typing import List, Dict, Set, FrozenSet, Tuple
from copy import deepcopy
import numpy as np


class GraphNP:
    """
    This class represents a Graph
    """

    def __init__(self, num_vertices: int = 0, edges: np.ndarray = None):
        """
        :param neighbours_mtx: a list of the edges in the graph - (u,v)
        :param vertex_array: a list of the vertices in the graph
        :param neighbors: a dictionary where the key is the number of a vertex, and the value is a set of vertex
         neighbors
        """
        self._vertices = np.arange(num_vertices)
        self._neighbours_mtx = None
        self._num_vertices = num_vertices
        self._num_edges = 0
        if edges is not None:
            self._neighbours_mtx = np.zeros((num_vertices, num_vertices))
            edges = edges.transpose()
            self._neighbours_mtx[edges[0], edges[1]] = True
            self._neighbours_mtx[edges[1], edges[0]] = True
            self._neighbours_mtx[self._vertices, self._vertices] = False
            self._num_edges = np.sum(self._neighbours_mtx) / 2

    def get_neighbours_mtx(self):
        return deepcopy(self._neighbours_mtx)
        # return self._neighbours_mtx

    def get_vertices(self):
        return self._vertices

    def set_neighbours_mtx(self, new_edges: np.ndarray):
        self._neighbours_mtx = new_edges

    def set_num_vertices(self, num_new_vertices: int):
        self._vertices = np.arange(num_new_vertices)
        self._num_vertices = num_new_vertices

    def get_num_vertices(self):
        return self._num_vertices

    def get_num_edges(self):
        return self._num_edges

    def __str__(self):
        return "Vertices: " + str(self._vertices) + "\nEdges Matrix: \n" + str(self._neighbours_mtx) + "\n"

    def draw_vertex_cover(self, vertex_cover):
        """
        This function draws the graph such that vertices that are part of the vertex cover are colored in green,
        otherwise in teal
        :param vertex_cover: a vertex cover
        :return:
        """
        G = nx.Graph()
        for vertex in self._vertices:
            if vertex in vertex_cover:
                G.add_node(vertex, color='green')
            else:
                G.add_node(vertex, color='red')

        for i in range(self._neighbours_mtx.shape[0]):
            for j in range(i + 1, self._neighbours_mtx.shape[0]):
                if self._neighbours_mtx[i, j]:
                    G.add_edge(i, j)
        colors = [node[1]['color'] for node in G.nodes(data=True)]
        nx.draw(G, node_color=colors, with_labels=True)
        plt.show()

    def create_p_random_graph(self, num_vertices: int, p: float):
        vertices = np.arange(num_vertices)
        self._vertices = vertices
        self._num_vertices = num_vertices
        edges = np.random.binomial(1, p, (num_vertices, num_vertices))
        edges[self._vertices, self._vertices] = False
        i_lower = np.tril_indices(num_vertices, -1)
        edges[i_lower] = edges.T[i_lower]
        self._neighbours_mtx = edges
        self._num_edges = np.sum(edges) / 2
