import numpy as np

# abc = None
# d = np.zeros((3, 3))
#
# a = np.array([[1, 0], [2, 1], [0, 2], [0,0], [1, 1]])
# b = a.transpose()
# print(d)
# d[b[0], b[1]] = True
# d[b[1], b[0]] = True
# print(d)
#
# d[np.arange(3), np.arange(3)] = False
# print(d)
#
# num_vertices = 5
# p = 0.5
# print()
# edges = np.random.binomial(1, p, (num_vertices, num_vertices))
# print(edges)
# # final_edges = np.zeros((num_vertices, num_vertices))
# edges[np.arange(num_vertices), np.arange(num_vertices)] = False
# i_lower = np.tril_indices(num_vertices, -1)
# edges[i_lower] = edges.T[i_lower]
# print(edges)

# prob_array = np.random.binomial(1, 0.9, (5,))
#
# print(prob_array)

# di = {1: 2, 2: 5, 3: 10, 4: 0, 5:10.1}
#
# print(max(di, key=di.get))

arr = np.array([1,2,3,10,11,12,13,14,15])
#
# print(np.random.choice(arr, 5, replace=False))
#
see = {10,2,3, 103, 4,0, 1, 24, 2, 3,4, 1}
print(np.array(list(see)))

# print(arr[np.array([0])])
#
#
# rand_vertices = np.array([0, 1, 4, 7, 10])
# state = np.array([0,1,1,0,0,0,1,1,1,0,1])
# #
# #
# # print(np.count_nonzero(arr2[arr1]))
#
#
# arr1 = np.array([0.2, 0.3, 0.01, 1, 0])
#
# prob_array = np.random.binomial(1, arr1)
#
# state[rand_vertices] = np.where(prob_array, np.logical_not(state[rand_vertices]), state[rand_vertices])
#
# print(prob_array)
# print(state)


