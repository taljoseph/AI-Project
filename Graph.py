import networkx as nx
import random
import matplotlib.pyplot as plt
from typing import List, Dict, Set, FrozenSet, Tuple
from copy import deepcopy


class Graph:
    """
    This class represents a Graph
    """

    def __init__(self, edges_list: List[FrozenSet[int]] = None,
                 vertex_list: List[int] = None,
                 neighbors: Dict[int, Set[int]] = None):
        """
        :param edges_list: a list of the edges in the graph - (u,v)
        :param vertex_list: a list of the vertices in the graph
        :param neighbors: a dictionary where the key is the number of a vertex, and the value is a set of vertex
         neighbors
        """
        self._edges = edges_list
        self._num_edges = 0 if edges_list is None else len(edges_list)
        self._vertices = vertex_list
        self._num_vertices = 0 if vertex_list is None else len(vertex_list)
        self._neighbors = neighbors

    def get_edges(self):
        return deepcopy(self._edges)

    def get_vertices(self):
        return self._vertices

    def set_edges(self, new_edges: List[FrozenSet[int]]):
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

    def __str__(self):
        return "Vertices: " + str(self._vertices) + "\nEdges: " + str(self._edges) + "\nNeighbors: " + \
               str(self._neighbors)

    def draw_vertex_cover(self, vertex_cover):
        """
        This function draws the graph1 such that vertices that are part of the vertex cover are colored in green,
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

        for edge in self._edges:
            e = list(edge)
            G.add_edge(e[0], e[1])
        colors = [node[1]['color'] for node in G.nodes(data=True)]
        pos = nx.circular_layout(G)
        nx.draw(G, node_color=colors, with_labels=True, pos=pos)
        plt.show()

    def create_p_random_graph(self, num_vertices: int, p: float):
        vertices = [i for i in range(num_vertices)]
        self._vertices = vertices
        self._num_vertices = num_vertices
        edges = []
        neighbors = {}
        for i in range(num_vertices):
            neighbors[i] = set()
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                if random.random() < p:
                    edges.append(frozenset({i, j}))
                    neighbors[i].add(j)
                    neighbors[j].add(i)
        self._edges = edges
        self._num_edges = len(edges)
        self._neighbors = neighbors

    def create_graph(self, num_vertices: int, edges: List[Tuple[int, int]]):
        vertices = [i for i in range(num_vertices)]
        new_edges = []
        neighbors = {}
        for v in vertices:
            neighbors[v] = set()
        for edge in edges:
            neighbors[edge[0]].add(edge[1])
            neighbors[edge[1]].add(edge[0])
            new_edges.append(frozenset({edge[0], edge[1]}))
        self._vertices = vertices
        self._num_vertices = num_vertices
        self._edges = new_edges
        self._num_edges = len(new_edges)
        self._neighbors = neighbors
