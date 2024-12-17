import scipy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import diags
import graphfunctions as fun
import random

order = 500
offset = 120
p = 0.2 #probability that an edge from the model graph is in the random graph

G = nx.circulant_graph(order,[i for i in range(1,offset+1)])
A = nx.laplacian_matrix(G).astype(float)

rows, cols = A.nonzero()
values = A[rows, cols]

# Combine indices and values into a list of tuples
nze = list(zip(rows, cols, values)) #nonzero elements

data = []
removed = []
for i in range(len(nze)):
    if int(nze[i][2]) == -1:
        if np.random.uniform(0,1) < 1 - p:
            data.append((nze[i][0], nze[i][1], 0))
            removed.append(nze[i])
        else:
            data.append(nze[i])
    else:
        data.append(nze[i])

rows, cols, values = zip(*data)

n_rows, n_cols = A.shape

# Create the sparse matrix laplacian matrix with removed edges
sparse_matrix = coo_matrix((values, (rows, cols)), shape=(n_rows, n_cols)) #coo format
L = sparse_matrix.tocsr() #convert

degrees = [2*offset for _ in range(order)]
degree_matrix = diags(degrees)  # create a diagonal matrix D

# Step 2: Compute the adjacency matrix A
M = degree_matrix - L  # A = D - L

# Convert to sparse adjacency matrix for efficiency
adjacency_matrix = csr_matrix(M)

#convert to networkx graph
newG = nx.Graph(nx.convert_matrix.from_scipy_sparse_array(adjacency_matrix))
print("old size", G.size())
print("new size", newG.size())
#fun.draw_graph(G)
fun.draw_graph(newG)
fun.plot_unscaled_eigenvectors(newG)
