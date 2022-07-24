import numpy as np

abc = None
d = np.zeros((3, 3))

a = np.array([[1, 0], [2, 1], [0, 2], [0,0], [1, 1]])
b = a.transpose()
print(d)
d[b[0], b[1]] = True
d[b[1], b[0]] = True
print(d)

d[np.arange(3), np.arange(3)] = False
print(d)

num_vertices = 5
p = 0.5
print()
edges = np.random.binomial(1, p, (num_vertices, num_vertices))
print(edges)
# final_edges = np.zeros((num_vertices, num_vertices))
edges[np.arange(num_vertices), np.arange(num_vertices)] = False
i_lower = np.tril_indices(num_vertices, -1)
edges[i_lower] = edges.T[i_lower]
print(edges)
