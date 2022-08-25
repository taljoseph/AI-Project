import networkx as nx
import random
import matplotlib.pyplot as plt
from typing import List, Dict, Set, FrozenSet, Tuple
from math import factorial


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
         neighbours
        """
        self._edges = edges_list
        self._num_edges = 0 if edges_list is None else len(edges_list)
        self._vertices = vertex_list
        self._num_vertices = 0 if vertex_list is None else len(vertex_list)
        self._neighbors = neighbors

    def get_edges(self):
        """
        :return: graph's edges.
        """
        return self._edges

    def get_vertices(self):
        """
        :return: graph's vertices
        """
        return self._vertices

    def set_edges(self, new_edges: List[FrozenSet[int]]):
        """
        Set graph's edges.
        :param new_edges: new edges.
        """
        self._edges = new_edges
        self._num_edges = len(new_edges)

    def set_vertices(self, new_vertices: List[int]):
        """
        Set graph's vertices.
        :param new_vertices: new vertices.
        """
        self._vertices = new_vertices
        self._num_vertices = len(new_vertices)

    def get_neighbors(self):
        """
        :return: Returns the neighbours - {vertex: Set[vertices]}
        """
        return self._neighbors

    def set_neighbors(self, new_neighbors: Dict[int, List[int]]):
        """
        Sets the neighbours.
        :param new_neighbors: new neighbours.
        """
        self._neighbors = new_neighbors

    def get_num_vertices(self):
        """
        :return: Number of vertices in the graph.
        """
        return self._num_vertices

    def get_num_edges(self):
        """
        :return: Number of edges in the graph.
        """
        return self._num_edges

    def __str__(self):
        """
        :return: Graph's string representation - vertices, edges, neighbours
        """
        return "Vertices: " + str(self._vertices) + "\nEdges: " + str(self._edges) + "\nNeighbors: " + \
               str(self._neighbors)

    def draw_vertex_cover(self, vertex_cover):
        """
        This function draws the graph1 such that vertices that are part of the vertex cover are colored in green,
        otherwise in teal
        :param vertex_cover: a vertex cover
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
        """
        Updates the graph to a p-random graph with the given number of vertices and probability 0 <= p <= 1.
        Each possible edge is added to the graph with probability p.
        :param num_vertices: Number of vertices in the new graph.
        :param p: Probability of each edge to be added.
        """
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

    def create_nx_graph(self, num_vertices: int, num_edges: int):
        """
        Updates the graph to a random graph with the number of vertices and edges given.
        :param num_vertices: Number of vertices.
        :param num_edges: Number of edges.
        """
        g = nx.gnm_random_graph(num_vertices, num_edges)
        self.create_graph(num_vertices, g.edges)

    def create_graph(self, num_vertices: int, edges: List[Tuple[int, int]]):
        """
        Updates graph to a graph with given number of vertices and list of edges.
        Converts the edges to the right format and creates the neighbours dictionary.
        :param num_vertices: Number of vertices in the new graph.
        :param edges: List of edges (tuples) in the new graph.
        """
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

    def create_bad_greedy_graph(self, k: int):
        """
        updates the graph to a graph which is bad for Greedy Hill Climbing algorithm (especially for a deterministic approach),
        as explained in the report.
        :param k: Integer > 0 for the graph creation.
        """
        factorial_k = factorial(k)
        vertices_lists = []
        num_vertices = 0
        l = 0
        while l < k:
            vertices_lists.append([i for i in range(num_vertices, int(factorial_k / (k - l)) + num_vertices)])
            num_vertices += len(vertices_lists[-1])
            l += 1
        upper_vertices = [i for i in range(num_vertices, factorial_k + num_vertices)]
        num_vertices += factorial_k
        edges = []
        l = 1
        while l <= k:
            i = 0
            j = -1
            while i < factorial_k:
                if i % l == 0:
                    j += 1
                edges.append((vertices_lists[k - l][j], upper_vertices[i]))
                i += 1
            l += 1
        self.create_graph(num_vertices, edges)

    def create_old_city_graph(self):
        num_vertices = 262
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 3),
                 (9, 13), (13, 14), (14, 15), (15, 16), (15, 17), (17, 18), (18, 19), (19, 20), (14, 21), (21, 22), (22, 23),
                 (23, 24), (24, 25), (25, 26), (25, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (32, 34),
                 (34, 35), (35, 13), (31, 36), (36, 10), (36, 37), (37, 38), (37, 39), (37, 40), (40, 41), (40, 42), (42, 31),
                 (23, 33), (29, 33), (42, 43), (43, 44), (44, 45), (45, 30), (44, 46), (46, 47), (47, 48), (48, 49), (49, 29),
                 (49, 50), (50, 51), (51, 28), (50, 52), (52, 53), (52, 54), (54, 55), (55, 48), (55, 56), (56, 57), (57, 58),
                 (58, 56), (58, 59), (59, 60), (60, 61), (60, 62), (62, 63), (63, 56), (63, 64), (64, 65), (65, 66), (66, 67),
                 (67, 68), (68, 69), (69, 43), (64, 43), (65, 70), (70, 71), (71, 72), (71, 73), (73, 74), (74, 75), (74, 76),
                 (76, 77), (77, 78), (77, 79), (79, 80), (80, 81), (81, 82), (82, 83), (83, 84), (83, 85), (85, 86), (85, 87),
                 (87, 88), (88, 89), (89, 90), (90, 91), (91, 92), (92, 93), (92, 89), (92, 88), (92, 94), (94, 95), (95, 96),
                 (96, 91), (96, 97), (97, 98), (98, 99), (99, 80), (99, 198), (96, 100), (100, 71), (100, 101), (101, 102),
                 (102, 103), (103, 104), (104, 105), (100, 106), (106, 107), (107, 70), (106, 59), (106, 108), (108, 109),
                 (109, 110), (110, 111), (111, 112), (109, 113), (113, 114), (114, 115), (115, 116), (116, 117), (117, 118),
                 (118, 58), (115, 119), (119, 120), (120, 121), (121, 122), (122, 123), (123, 124), (123, 125), (125, 109),
                 (125, 126), (126, 127), (127, 128), (128, 129), (129, 130), (130, 131), (131, 132), (132, 95), (132, 133),
                 (133, 134), (134, 135), (135, 136), (136, 137), (137, 138), (136, 94), (138, 135), (138, 139), (139, 134),
                 (139, 140), (140, 133), (140, 134), (140, 141), (141, 131), (141, 142), (142, 140), (142, 143), (143, 140),
                 (137, 144), (144, 145), (145, 139), (145, 138), (145, 146), (146, 129), (146, 147), (147, 148), (148, 149),
                 (149, 150), (150, 151), (151, 152), (152, 153), (153, 154), (154, 128), (154, 155), (155, 156), (156, 157),
                 (157, 158), (157, 127), (155, 159), (159, 160), (159, 161), (161, 162), (162, 163), (163, 164), (163, 165),
                 (165, 166), (166, 167), (167, 168), (167, 169), (169, 153), (169, 170), (170, 171), (171, 172), (171, 173),
                 (173, 174), (171, 174), (174, 175), (175, 176), (176, 177), (177, 178), (178, 179), (179, 180), (179, 181),
                 (179, 182), (182, 183), (183, 184), (184, 185), (185, 186), (186, 187), (187, 188), (188, 189), (189, 190),
                 (190, 181), (190, 191), (190, 192), (192, 193), (193, 194), (194, 195), (195, 176), (193, 196), (196, 197),
                 (197, 199), (199, 200), (200, 201), (201, 202), (202, 144), (201, 203), (203, 204), (204, 205),
                 (205, 196), (205, 206), (206, 191), (206, 185), (204, 207), (207, 208), (208, 89), (208, 209), (209, 210),
                 (210, 88), (210, 211), (211, 212), (212, 213), (213, 214), (214, 87), (214, 215), (215, 216), (216, 217),
                 (217, 218), (218, 219), (219, 215), (219, 220), (220, 221), (221, 222), (222, 223), (223, 224), (222, 225),
                 (221, 226), (226, 227), (227, 228), (228, 229), (227, 230), (230, 231), (231, 226), (231, 232), (232, 233),
                 (233, 230), (233, 234), (234, 230), (234, 235), (235, 236), (236, 220), (235, 237), (237, 238), (238, 239),
                 (239, 240), (240, 241), (241, 242), (242, 243), (243, 239), (239, 209), (241, 213), (242, 212), (243, 211),
                 (239, 244), (244, 207), (244, 245), (245, 205), (245, 238), (245, 246), (246, 238), (246, 237), (246, 247),
                 (247, 248), (248, 249), (249, 250), (250, 251), (251, 233), (251, 252), (252, 253), (253, 245), (253, 249),
                 (254, 252), (254, 251), (254, 185), (254, 184), (254, 255), (255, 233), (255, 256), (256, 257), (257, 258),
                 (258, 259), (259, 260), (260, 261), (258, 260), (73, 79), (198, 97), (108, 126), (151, 173), (203, 94), (222, 218)]
        self.create_graph(num_vertices, edges)
