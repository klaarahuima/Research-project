import networkx as nx
import graphfunctions as fun
import duplicatingfunctions as dup
import knpaths as path
from graphfunctions import get_adj_eigenvalues, get_adjacency_eigenvectors
import circulants as cry
import removingedges as rem
import matplotlib.pyplot as plt
import numpy as np


n = 1000
d = 20 #per side
p = 1 # probability that an edge IS in the graph
"""
G = rem.simulate(n, d, 0.9)
H = rem.simulate(n, d, 0.5)
J = rem.simulate(n, d, 0.3)
"""

G = fun.random_graph_unit_circle(order = n, threshold = 10, p = 1, q = 0, likelihood = 2)
H = fun.random_graph_unit_circle(order = n, threshold = 10, p = 0.8, q = 0, likelihood = 2)
J = fun.random_graph_unit_circle(order = n, threshold = 10, p = 0.4, q = 0, likelihood = 2)
fun.add_eigenvectors(G, 0, 1)
fun.add_eigenvectors(H, 0, 1)
fun.add_eigenvectors(J, 0, 1)

print(fun.error(G, type="lap"))
print(fun.error(H, type="lap"))
print(fun.error(J, type="lap"))


fun.plot_unscaled_eigenvectors(G)
fun.plot_unscaled_eigenvectors(H)
fun.plot_unscaled_eigenvectors(J)







