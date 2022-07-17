import networkx as nx
import random
import matplotlib.pyplot as plt


class Graph:
    """
    This class represents a Graph
    """

    def __init__(self, edges_list, num_edges, vertex_list, num_vertices, neighbors, vertex_edges):
        """
        :param edges_list: a list of the edges in the graph - (u,v)
        :param num_edges: the number of edges in the graph
        :param vertex_list: a list of the vertices in the graph
        :param num_vertices: the number of vertices in the graph
        :param neighbors: a dictionary where the key is the number of a vertex, and the value is a list of vertex
         neighbors
        :param vertex_edges: a dictionary where the key is the number of a vertex, and the value is a list of edges
         that are associated to that vertex
        """
        self.edges_list = edges_list
        self.num_edges = num_edges
        self.vertex_list = vertex_list
        self.num_vertices = num_vertices
        self.neighbors = neighbors
        self.vertex_edges = vertex_edges

    def two_approximate_vertex_cover(self):
        """
        This function finds a vertex cover using the 2-approximation algorithm
        :return: a list of vertices that are a vertex cover
        """
        vertex_cover = []
        edges_list = self.edges_list[:]
        edges_removed = set()
        while edges_list:
            edge = random.choice(edges_list)
            vertex_cover.append(edge[0])
            vertex_cover.append(edge[1])
            edges_to_remove = list(set(self.vertex_edges[edge[0]] + self.vertex_edges[edge[1]]).difference(edges_removed))
            for e in edges_to_remove:
                edges_list.remove(e)
            edges_removed = edges_removed.union(edges_to_remove)
        return vertex_cover

    def draw_vertex_cover(self, vertex_cover):
        """
        This function draws the graph such that vertices that are part of the vertex cover are colored in green,
        otherwise in teal
        :param vertex_cover: a vertex cover
        :return:
        """
        G = nx.Graph()
        for vertex in self.vertex_list:
            if vertex in vertex_cover:
                G.add_node(vertex, color='green')
            else:
                G.add_node(vertex, color='teal')

        for edge in self.edges_list:
            G.add_edge(edge[0], edge[1])
        colors = [node[1]['color'] for node in G.nodes(data=True)]
        nx.draw(G, node_color=colors, with_labels=True)
        plt.show()

