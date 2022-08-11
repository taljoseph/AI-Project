from abc import abstractmethod, ABC
from Graph import *
import numpy as np
import math

def softmax(fitness_array: np.ndarray):
    fitness_array -= fitness_array.max() + 1
    fitness_array = np.exp(fitness_array)
    fitness_array /= fitness_array.sum()
    return fitness_array


class VC_GA(ABC):
    def __init__(self, graph: Graph):
        self._graph = graph

    @abstractmethod
    def fitness(self, state: np.ndarray):
        pass

    @abstractmethod
    def reproduce(self, state1: np.ndarray, state2: np.ndarray, s1_fitness: float, s2_fitness: float):
        pass

    @abstractmethod
    def mutation(self, state: np.ndarray):
        pass

    def create_n_random_states(self, n: int) -> np.ndarray:
        return np.random.binomial(1, 0.5, (n, self._graph.get_num_vertices()))

    def perform_ga(self, num_gens: int, population_size: int):
        # TODO best vc or best fitness, currently best fitness
        states = self.create_n_random_states(population_size)

        fitness_array = np.zeros(population_size)
        for i in range(population_size):
            fitness_array[i] = self.fitness(states[i])
        best_sol = states[fitness_array.argmax()]
        best_sol_val = fitness_array.max()

        for i in range(num_gens):
            probs = softmax(fitness_array)
            rand_pairs = np.random.choice(np.arange(population_size), size=(population_size, 2), p=probs)
            copy = states.copy()
            for j in range(population_size):
                first_state_arg = rand_pairs[j][0]
                second_state_arg = rand_pairs[j][1]
                states[j] = self.mutation(self.reproduce(copy[first_state_arg], copy[second_state_arg],
                                                             fitness_array[first_state_arg], fitness_array[second_state_arg]))
            for j in range(population_size):
                fitness_array[j] = self.fitness(states[j])

            best_fitness_arg = fitness_array.argmax()
            if best_sol_val < fitness_array[best_fitness_arg]:
                best_sol_val = fitness_array[best_fitness_arg]
                best_sol = states[best_fitness_arg].copy()

        return np.flatnonzero(best_sol).tolist()


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

class RegularVC_GA2(VC_GA):
    def __init__(self, graph):
        super(RegularVC_GA2, self).__init__(graph)
        vertices = self._graph.get_vertices()
        neighbours = graph.get_neighbors()
        self._vertex_edges = {v: {frozenset({v, u}) for u in neighbours[v]} for v in vertices}

    def fitness(self, state: np.ndarray):
        edges_covered = set()
        vertices = np.flatnonzero(state)
        for v in vertices:
            edges_covered |= self._vertex_edges[v]
        return len(edges_covered) + state.size - vertices.size

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


class RegularVC_GA3(VC_GA):
    def __init__(self, graph):
        super(RegularVC_GA3, self).__init__(graph)
        vertices = self._graph.get_vertices()
        neighbours = graph.get_neighbors()
        self._vertex_edges = {v: {frozenset({v, u}) for u in neighbours[v]} for v in vertices}

    def fitness(self, state: np.ndarray):
        edges_covered = set()
        vertices = np.flatnonzero(state)
        for v in vertices:
            edges_covered |= self._vertex_edges[v]
        return len(edges_covered) + state.size - vertices.size

    def reproduce(self, state1: np.ndarray, state2: np.ndarray, s1_fitness: float, s2_fitness: float):
        ind = state1.size // 2
        return np.concatenate((state1[:ind], state2[ind:]))

    def mutation(self, state: np.ndarray):
        num_vertices = self._graph.get_num_vertices()
        prob_array = np.random.binomial(1, 1/num_vertices, (num_vertices,))
        return np.where(prob_array, np.logical_not(state), state)



class VCPunish_GA(VC_GA):
    def __init__(self, graph):
        super(VCPunish_GA, self).__init__(graph)
        vertices = self._graph.get_vertices()
        neighbours = graph.get_neighbors()
        self._vertex_edges = {v: {frozenset({v, u}) for u in neighbours[v]} for v in vertices}

    def fitness(self, state: np.ndarray):
        edges_covered = set()
        vertices = np.flatnonzero(state)
        for v in vertices:
            edges_covered |= self._vertex_edges[v]
        punishment = 0 if len(edges_covered) == self._graph.get_num_edges() else 5
        return len(edges_covered) - vertices.size - punishment

    def reproduce(self, state1: np.ndarray, state2: np.ndarray, s1_fitness: float, s2_fitness: float):
        p = 0.5
        if s1_fitness or s2_fitness:
            p = s1_fitness / (s1_fitness + s2_fitness)  # probability to choose vertex from first state
        num_vertices = self._graph.get_num_vertices()
        prob_array = np.random.binomial(1, p, (num_vertices,))
        return np.where(prob_array, state1, state2)

    def mutation(self, state: np.ndarray):
        p = random.random()
        if p < 0.7:
            num_vertices = self._graph.get_num_vertices()
            prob_array = np.random.binomial(1, 2/num_vertices, (num_vertices,))
            return np.where(prob_array, np.logical_not(state), state)
        else:
            edges = set(self._graph.get_edges())
            edges_covered = set()
            vertices = np.flatnonzero(state)
            for v in vertices:
                edges_covered |= self._vertex_edges[v]
            edges_not_covered = edges - edges_covered
            edges_not_covered = list(edges_not_covered)
            if edges_not_covered:
                rand_edge = np.random.choice(np.arange(len(edges_not_covered)), size=1)[0]
                # a = edges_not_covered[rand_edge[0]]
                ver = random.sample(edges_not_covered[rand_edge], 1)[0]
                # a = random.sample(edges_not_covered[3], 1)[0]
                # for i in rand_edges:
                    # a = edges_not_covered[i]
                    # ver.add(random.sample(edges_not_covered[i], 1)[0])
                # ver = np.array(list(ver))
                # rand_v = np.random.choice(np.flatnonzero(state), size=len(ver), replace=False)
                # state[rand_v] = 0
                state[ver] = 1
            # negs = np.argwhere(state == False).flatten()
            # change = np.random.choice(negs, size=1, replace=False)
            # state[change] = 1
            # return state
        return state


class FixedSizeVCGA(VC_GA):
    def __init__(self, graph, k):
        super(FixedSizeVCGA, self).__init__(graph)
        vertices = self._graph.get_vertices()
        neighbours = graph.get_neighbors()
        self._vertex_edges = {v: {frozenset({v, u}) for u in neighbours[v]} for v in vertices}
        self._k = k

    def get_k(self):
        return self._k

    def update_k(self, k):
        self._k = k

    def create_n_random_states(self, n: int) -> np.ndarray:
        num_vertices = self._graph.get_num_vertices()
        samples = np.zeros((n, num_vertices), dtype=int)
        samples[:, 0:self._k] = 1
        rng = np.random.default_rng()
        rng.permuted(samples, axis=1, out=samples)
        return samples

    def fitness(self, state: np.ndarray):
        edges_covered = set()
        vertices = np.flatnonzero(state)
        for v in vertices:
            edges_covered |= self._vertex_edges[v]
        return len(edges_covered)

    def reproduce(self, state1: np.ndarray, state2: np.ndarray, s1_fitness: float, s2_fitness: float):
        combined_vertices = np.argwhere(state1 | state2).flatten()
        chosen_vertices = np.random.choice(combined_vertices, size=self._k, replace=False)
        res = np.zeros(self._graph.get_num_vertices())
        res[chosen_vertices] = 1
        return res

    def mutation(self, state: np.ndarray):
        num_vertices = self._k // 10 + 1
        pos = np.random.choice(np.flatnonzero(state), size=num_vertices, replace=False)
        neg = np.random.choice(np.argwhere(state == False).flatten(), size=num_vertices, replace=False)
        state[pos] = 0
        state[neg] = 1
        # edges = set(self._graph.get_edges())
        # edges_covered = set()
        # vertices = np.flatnonzero(state)
        # for v in vertices:
        #     edges_covered |= self._vertex_edges[v]
        # edges_not_covered = edges - edges_covered
        # edges_not_covered = list(edges_not_covered)
        # if edges_not_covered:
        #     rand_edges = np.random.choice(np.arange(len(edges_not_covered)), size=5)
        #     ver = set()
        #     for i in rand_edges:
        #         # a = edges_not_covered[i]
        #         ver.add(random.sample(edges_not_covered[i], 1)[0])
        #     ver = np.array(list(ver))
        #     rand_v = np.random.choice(np.flatnonzero(state), size=len(ver), replace=False)
        #     state[rand_v] = 0
        #     state[ver] = 1

        return state


class VC_NEW_MUT(VC_GA):
    def __init__(self, graph):
        super(VC_NEW_MUT, self).__init__(graph)
        self._vertices = self._graph.get_vertices()
        self._neighbours = graph.get_neighbors()
        self._vertex_edges = {v: {frozenset({v, u}) for u in self._neighbours[v]} for v in self._vertices}

    def fitness(self, state: np.ndarray):
        edges_covered = set()
        vertices = np.flatnonzero(state)
        for v in vertices:
            edges_covered |= self._vertex_edges[v]
        return len(edges_covered) - vertices.size

    def reproduce(self, state1: np.ndarray, state2: np.ndarray, s1_fitness: float, s2_fitness: float):
        p = 0.5
        if s1_fitness or s2_fitness:
            p = s1_fitness / (s1_fitness + s2_fitness)  # probability to choose vertex from first state
        num_vertices = self._graph.get_num_vertices()
        prob_array = np.random.binomial(1, p, (num_vertices,))
        return np.where(prob_array, state1, state2)

    def mutation(self, state: np.ndarray):
        rand_vertices = np.random.choice(self._vertices, math.ceil(0.01 * len(self._vertices)), replace=False)
        prob_array = np.zeros(rand_vertices.size)
        for i in range(prob_array.size):
            neighbours = np.array(list(self._neighbours[rand_vertices[i]]))
            if neighbours.size == 0:
                prob_array[i] = state[rand_vertices[i]]
            else:
                prob_array[i] = np.count_nonzero(state[neighbours])
                if state[rand_vertices[i]] == 0:
                    prob_array[i] = neighbours.size - prob_array[i]
                prob_array[i] /= neighbours.size
        prob_array = np.random.binomial(1, prob_array)
        state[rand_vertices] = np.where(prob_array, np.logical_not(state[rand_vertices]), state[rand_vertices])
        return state
