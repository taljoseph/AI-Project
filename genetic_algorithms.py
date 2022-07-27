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
    def reproduce(self, state1: np.ndarray, state2: np.ndarray):
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
        best_sol = states[fitness_array.argmax()]
        best_sol_val = fitness_array.max()

        for i in range(num_gens):
            selection_arr = fitness_array / fitness_array.sum()
            # selection_arr = self.selection(states)
            rand_states_pairs = states[np.random.choice(np.arange(population_size), size=(population_size, 2), p=selection_arr)]
            for i in range(population_size):
                states[i] = self.mutation(self.reproduce(rand_states_pairs[i][0], rand_states_pairs[i][1]))
                fitness_array[i] = self.fitness(states[i])

            best_fitness_arg = fitness_array.argmax()
            if best_sol_val < fitness_array[best_fitness_arg]:
                best_sol_val = fitness_array[best_fitness_arg]
                best_sol = states[best_fitness_arg]

        return best_sol
