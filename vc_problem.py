from Graph import *


class VC_Problem:
    """
    Class representing a Vertex Cover problem
    """

    def __init__(self, graph: Graph, initial_state: List[int] = None):
        self._graph = graph
        self._initial_state = initial_state if initial_state is not None else self._create_random_initial_state()

    def _create_random_initial_state(self):
        """
        creates and returns a random initial state - random subsequence of the
        vertices in the graph
        """
        num_vertices_graph = self._graph.get_num_vertices()
        num_vertices_vc = random.randint(0, num_vertices_graph)
        return random.sample([i for i in range(num_vertices_graph)], num_vertices_vc)

    def get_initial_state(self):
        """
        returns the initial state
        :return: list of vertices
        """
        return self._initial_state

    def get_neighbors(self, state: List[int]) -> List[List[int]]:
        """
        Returns a List of neighboring states. A neighbor of a state is the state
        excluding one of the nodes it currently has or an addition of a node
        it currently does not have
        :param state: list of vertices
        :return: List of neighboring states
        """
        vertices = self._graph.get_vertices()
        neighbors = []
        for i in range(len(state)):
            neighbors.append(state[:i] + state[i + 1:])
        return neighbors + [state + [v] for v in vertices if v not in state]

    def is_goal_state(self, state: List[int]) -> bool:
        """
        Checks whether the state is a goal state i.e if the state represents
        a valid vertex cover.
        :param state: List of nodes
        :return:
        """
        edges = self._graph.get_edges()
        vc = set(state)
        for edge in edges:
            e = list(edge)
            if e[0] not in vc and e[1] not in vc:
                return False
        return True

    def get_graph(self):
        return self._graph




