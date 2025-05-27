import scipy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import diags
import graphfunctions as fun
import numpy as np
import random

"""
Functions defined to generate graphs in which we remove edges from perfect circulant graphs (like in the Richter Rocha paper).
I don't actually run this file, I call this function by importing this file into my main simulation file and calling the function
simulate defined below.
"""

# Defining parameters of graph:
d = 20 # number of nearest neighbours every vertex is connected to (per side)
n = 1000 # order of graph
p = 0.6 # probability two vertices that are neighbours in original circulant graph will be connected in derived random graph

# Simulating edge removal
def simulate(n, d, p):
    G = nx.circulant_graph(n, [i for i in range(d)]) # standard networkx function for generating circulant graph
    print(G)
    H = G.copy() # creating copy of graph
    for u,v in G.edges: # iterating over every edge in graph
        if np.random.uniform(0, 1) < 1 - p: # removing edges uniformly with probability 1 -p
            H.remove_edge(u,v)
    print(H)
    return H # return final random graph

