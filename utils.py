# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from genetic_algorithms import *
import glob
import pandas as pd


def create_random_initial_state(graph: Graph):
    """
    creates and returns a random initial state - random subsequence of the
    vertices in the graph
    @param graph: A Graph object
    @return: A random initial state
    """
    num_vertices_vc = random.randint(0, graph.get_num_vertices())
    return random.sample(graph.get_vertices(), num_vertices_vc)


def is_vc(graph: Graph, state: List[int]) -> bool:
    """
    This function checks if the state is the goal state - a vertex cover
    @param graph: The Graph object in question
    @param state: The state to check if it represents a goal state in the graph
    @return: True if goal state, otherwise false
    """
    edges = graph.get_edges()
    vc = set(state)
    for edge in edges:
        e = list(edge)
        if e[0] not in vc and e[1] not in vc:
            return False
    return True


def get_edges_covered_by_vertex(v: int, neighbours: Dict[int, Set[int]]) -> Set[FrozenSet[int]]:
    """
    This function returns a set of all edges covered by vertex v
    @param v: The vertex in question
    @param neighbours: The neighbours dictionary representing the neighbours of every vertex
    @return: The edges covered by vertex v
    """
    edges_covered = set()
    for u in neighbours[v]:
        edges_covered.add(frozenset({u, v}))
    return edges_covered


def get_edges_covered(state: List[int], graph: Graph) -> Set[FrozenSet[int]]:
    """
    This function returns a set of all edges covered by the current state
    @param state: The state in question
    @param graph: The graph object
    @return: A set of all edges in the graph covered in the state
    """
    edges_covered = set()
    for u in state:
        for u_neighbor in graph.get_neighbors()[u]:
            edges_covered.add(frozenset({u, u_neighbor}))
    return edges_covered


def build_graph_from_file(file_path):
    """
    Builds a graph from an mis file
    @param file_path: The path of the graph file
    @return: The graph created
    """
    file = open(file_path, "r")
    first_line = file.readline().split()
    num_of_vertices = int(first_line[2])
    edges = []
    for line in file:
        line = line.split()
        if line[0] == "e":
            edges.append((int(line[1]) - 1, int(line[2]) - 1))
    graph = Graph()
    graph.create_graph(num_of_vertices, edges)
    file.close()
    return graph


def create_random_graph_file(num_vertices, num_edges, path):
    """
    Creates an mis graph file for a random graph
    @param num_vertices: Number of vertices
    @param num_edges: Number of edges
    @param path: Path to save the file to
    """
    g = Graph()
    g.create_nx_graph(num_vertices, num_edges)
    create_graph_file(g, path)


def create_p_random_graph_file(num_vertices, p, path):
    """
    Creates an mis graph file for a p-random graph
    @param num_vertices: Number of vertices
    @param p: The probability for each edge
    @param path: Path to save the file to
    """
    g = Graph()
    g.create_p_random_graph(num_vertices, p)
    create_graph_file(g, path)


def create_graph_file(graph, path):
    """
    Creates an mis graph file
    @param graph: The graph to create a file for
    @param path: The path to save to
    """
    f = open(path, "w")
    f.write("p edge {} {}\n".format(graph.get_num_vertices(), graph.get_num_edges()))
    for edge in graph.get_edges():
        v, u = edge
        f.write("e {} {}\n".format(v + 1, u + 1))
    f.close()


def get_population_size(num_vertices):
    """
    Gets the recommended population size based on the number of vertices
    @param num_vertices: The number of vertices
    @return: The recommended population size
    """
    return round((0.315 * (num_vertices ** 0.6) + 0.72))


def algorithms_total_scores(path, column_name: str):
    """
    Calculates the scores of each algorithm over all results files
    @param path: The path of the directory containing the graphs
    @param column_name: The name of the column - Avg VC Size OR Min VC Size
    @return: The scores
    """
    files = glob.glob(path)
    sum_avg_values = np.zeros(4)
    for file in files:
        df = pd.read_excel(file)
        a = df[column_name]
        a = a.to_numpy()
        sum_avg_values += a
    return sum_avg_values
