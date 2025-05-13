import networkx as nx
import numpy as np
import graphfunctions as fun
import duplicatingfunctions as dup



def k_path_graph(n, k):
    G = nx.Graph()
    for i in range(n):
        for j in range(1, k + 1):
            if i + j < n:
                G.add_edge(i, i + j)
            if i - j >= 0:
                G.add_edge(i, i - j)
    return G


