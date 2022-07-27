from abc import abstractmethod, ABC
from Graph import *
import numpy as np


class VC_GA(ABC):
    def __init__(self, graph: Graph):
        self._graph = graph
        # self._vertices = np.array(graph.get_vertices())


    @abstractmethod
    def fitness(self, state: np.ndarray):
        pass

    @abstractmethod
    def reproduce(self, state1: np.ndarray, state2: np.ndarray, s1_fitness: float, s2_fitness: float):
        pass

    @abstractmethod
    def mutation(self, state: np.ndarray):
        pass

    def create_n_random_states(self, n: int) -> np.ndarray[np.ndarray]:
        return np.random.binomial(1, 0.5, (n, self._graph.get_num_vertices()))

    # def selection(self, states: np.ndarray[np.ndarray]):
    #     num_states = states.shape[0]
    #     fitness_array = np.zeros(num_states)
    #     for i in range(num_states):
    #         fitness_array[i] = self.fitness(states[i])
    #     return fitness_array / fitness_array.sum()

    def perform_ga(self, num_gens: int, population_size: int):
        # TODO best vc or best fitness, currently best fitness
        states = self.create_n_random_states(population_size)

        fitness_array = np.zeros(population_size)
        for i in range(population_size):
            fitness_array[i] = self.fitness(states[i])
        if fitness_array.min() < 0:
            fitness_array += abs(fitness_array.min())
        best_sol = states[fitness_array.argmax()]
        best_sol_val = fitness_array.max()

        for i in range(num_gens):
            selection_arr = fitness_array / fitness_array.sum()
            rand_pairs = np.random.choice(np.arange(population_size), size=(population_size, 2), p=selection_arr)
            for i in range(population_size):
                first_state_arg = rand_pairs[i][0]
                second_state_arg = rand_pairs[i][1]
                states[i] = self.mutation(self.reproduce(states[first_state_arg], states[second_state_arg],
                                                         fitness_array[first_state_arg], fitness_array[second_state_arg]))
            for i in range(population_size):
                fitness_array[i] = self.fitness(states[i])
            if fitness_array.min() < 0:
                fitness_array += abs(fitness_array.min())

            best_fitness_arg = fitness_array.argmax()
            if best_sol_val < fitness_array[best_fitness_arg]:
                best_sol_val = fitness_array[best_fitness_arg]
                best_sol = states[best_fitness_arg]

        return best_sol


class RegularVC_GA(VC_GA):
    def __init__(self, graph):
        super(RegularVC_GA, self).__init__(graph)
        vertices = self._graph.get_vertices()
        neighbours = graph.get_neighbors()
        self._vertex_edges = {v: {frozenset({v, u}) for u in neighbours[v]} for v in vertices}

    def fitness(self, state: np.ndarray):
        edges_covered = set()
        vertices = np.flatnonzero(state)
        for v in vertices:
            edges_covered |= self._vertex_edges[v]
        return 2 * len(edges_covered) - vertices.size

    def reproduce(self, state1: np.ndarray, state2: np.ndarray, s1_fitness: float, s2_fitness: float):
        p = 0.5
        if s1_fitness or s2_fitness:
            p = s1_fitness / (s1_fitness + s2_fitness)  # probability to choose vertex from first state
        num_vertices = self._graph.get_num_vertices()
        prob_array = np.random.binomial(1, p, (num_vertices,))
        return np.where(prob_array, state1, state2)

    def mutation(self, state: np.ndarray):
        num_vertices = self._graph.get_num_vertices()
        prob_array = np.random.binomial(1, 1/num_vertices, (num_vertices,))
        return np.where(prob_array, np.logical_not(state), state)
