import graphfunctions as fun
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.optimize import curve_fit

"""
Functions used to generate ideal circulant graphs with exactly twice as many vertices in 1st quadrant (not random).
I don't actually run this file, I call this function by importing this file into my main simulation file and calling the function
generate_graph and split_vectors defined below.
"""

def generate_graph(n):
    """
    Args:
        n: order of generated graph

    Returns: networkx graph of the model described above
    """
    G = nx.Graph() # create empty graph
    num_circulant = n//5*2 # number of vertices in 1st quadrant
    k = 2  # Number of closest neighbors on each side
    G.add_nodes_from([i for i in range(n)])
    # below: adding edges at the boundary between 1st quadrant and rest of graph
    G.add_edge(0, 1)
    G.add_edge(0,2)
    G.add_edge(1,3)
    G.add_edge(1, 2)
    for i in range(2,num_circulant-2): # adding all the other edges in 1st quadrant
        G.add_edge(i, (i + 1))  # Add edge to the right
        G.add_edge(i, (i - 1))  # Add edge to the left
        G.add_edge(i, (i - 2))  # Add edge to the left
        G.add_edge(i, (i + 2))  # Add edge to the left
    G.add_edge(num_circulant-2,num_circulant-1)

    for i in range(num_circulant,n-1): # adding edges to rest of graph
        G.add_edge(i, (i + 1))  # Add edge to the next vertex
        G.add_edge(i, (i - 1)) # Add edge to the previous vertex
    G.add_edge(n-1,0)
    fun.add_eigenvectors(G, 0, 1)

    return G




def split_vectors(G, num_circulant):
    """
    The function splits the 2nd and 3rd laplacian eigenvector of a networkx graph G into 2 parts:
    One with vertices in 1st quadrant, and one with vertices in remaining quadrants.
    The function ignores vertices on the boundary of the 1st quadrant to minimize the deviations it has on the vertices there.

    I used this to derive the constants of the sinusoidal functions that describe the eigenvectors of the
    ideal random circulant graph model.

    Args:
        G: graph
        num_circulant: number of vertices in 1st quadrant

    Returns: 4 vectors:
            s: 2nd laplacian eigenvector, 1st quadrant
            p: 2nd laplacian eigenvector, 2nd-4th quadrant
            r: 3rd laplacian eigenvector, 1st quadrant
            q: 3rd laplacian eigenvector, 2nd-4th quadrant

    """
    s = G.graph['second_lap'][2:num_circulant-2]
    p = G.graph['second_lap'][num_circulant:]
    r = G.graph['third_lap'][2:num_circulant-2]
    q = G.graph['third_lap'][num_circulant:]
    return (s, p ,r, q)
