import networkx as nx
import random
import matplotlib.pyplot as plt
from typing import List, Dict, Set, Tuple
from copy import deepcopy


class Graph:
    """
    This class represents a Graph
    """

    def __init__(self, edges_list: List[Set[int]], vertex_list: List[int], neighbors: Dict[int, Set[int]]):
        """
        :param edges_list: a list of the edges in the graph - (u,v)
        :param vertex_list: a list of the vertices in the graph
        :param neighbors: a dictionary where the key is the number of a vertex, and the value is a list of vertex
         neighbors
        """
        self._edges = edges_list
        self._num_edges = len(edges_list)
        self._vertices = vertex_list
        self._num_vertices = len(vertex_list)
        self._neighbors = neighbors

    def get_edges(self):
        return deepcopy(self._edges)

    def get_vertices(self):
        return self._vertices

    def set_edges(self, new_edges: List[Set[int]]):
        self._edges = new_edges

    def set_vertices(self, new_vertices: List[int]):
        self._vertices = new_vertices

    def get_neighbors(self):
        return deepcopy(self._neighbors)

    def set_neighbors(self, new_neighbors: Dict[int, List[int]]):
        self._neighbors = new_neighbors

    def get_num_vertices(self):
        return self._num_vertices

    def get_num_edges(self):
        return self._num_edges


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
                G.add_node(vertex, color='teal')

        for edge in self._edges:
            e = list(edge)
            G.add_edge(e[0], e[1])
        colors = [node[1]['color'] for node in G.nodes(data=True)]
        nx.draw(G, node_color=colors, with_labels=True)
        plt.show()

