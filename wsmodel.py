from scipy.sparse import csr_matrix

import graphfunctions as fun
import scipy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

order = 1000
offset = 100
p = 0.1  # Rewiring probability
G = nx.watts_strogatz_graph(order, offset, p)

#sparse_array = nx.to_scipy_sparse_array(G, format="csr")
#M = csr_matrix(sparse_array)

nx.draw(G, with_labels=False, node_size=3, node_color="blue", edge_color="lightgray")
plt.title("Watts-Strogatz Small-World Network")
plt.show()

fun.plot_unscaled_eigenvectors(G)
