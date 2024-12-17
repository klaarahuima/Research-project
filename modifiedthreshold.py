import numpy as np
import graphfunctions as fun
from graphfunctions import plot_degree_distribution

"Define parameters of graph"
order = 800
threshold_degree = 20 #max distance to 'near-by' vertices in degrees
p = 0.8 #probability of connecting two vertices within the threshold distance
q = 0.001 #probability of connecting two vertices no withing the threshold distance
threshold = np.deg2rad(threshold_degree)

"Choose what to display"
graph_networkx = False #networkx default visualisation
graph_circle = False #vertices and edges on the unit circle
embedding = True #embedding of 2nd and 3rd eigenvectors
vertex_distribution = False #histogram of how many vertices are in each quadrant
degree_distribution = False #degree histogram


G = fun.random_graph_unit_circle(order, threshold, p, q)

if graph_networkx:
    fun.draw_graph(G)
if graph_circle:
    fun.plot_graph_unit_circle(G, title=f"Random Graph (n={order}, p={p}, q={q}, threshold={np.rad2deg(threshold):.1f}°)")
if embedding:
    fun.plot_unscaled_eigenvectors(G)
if vertex_distribution:
    fun.vertex_histogram(G.graph['dictionary'])
if degree_distribution:
    fun.plot_degree_distribution(G)

avg_neighbourhood = order/360 * threshold_degree * 2 * p

print("size: ",G.size())
print("average number of neighbours within threshold degree: ", avg_neighbourhood)


