import graphfunctions as fun
import networkx as nx

G = nx.Graph()

n = 10
G.add_nodes_from([i for i in range(10)])
G.add_edges_from([(0,1), (0,2), (0,9), (1,2), (1,3), (1,9), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9)])

S = nx.circulant_graph(8, [1])
T = nx.circulant_graph(16, [1,2])

print('small ',nx.adjacency_matrix(S).astype(float))
print('large ',nx.adjacency_matrix(T).astype(float))
print('goal ',nx.adjacency_matrix(G).astype(float))

print('small cycle: ',fun.get_eigenvalues(S))
print('big circulant: ', fun.get_eigenvalues(T))
print('non-uniform: ', fun.get_eigenvalues(G))




