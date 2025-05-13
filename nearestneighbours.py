import numpy as np
import graphfunctions as fun

order = 600
max_inversions = order*(order-1)/2
neighbours = 50

"Choose what to display"
graph_networkx = False #networkx default visualisation
graph_circle = False #vertices and edges on the unit circle
embedding = True #embedding of 2nd and 3rd eigenvectors
vertex_distribution = False #histogram of how many vertices are in each quadrant
degree_distribution = False #degree histogram
error =True  #kendall distance of ordering of vertices
plot_inversions = True
plot_eigenvectors = True

G = fun.nearest_neighbours(order,neighbours)
#G = add.create_graph(75,25)


if graph_networkx:
    fun.draw_graph(G)
if graph_circle:
    fun.plot_graph_unit_circle(G, title=f"Random Graph")

if vertex_distribution:
    fun.vertex_histogram(G.graph['dictionary'])

if degree_distribution:
    avg = fun.plot_degree_distribution(G)
    print("average degree: ", avg)

if embedding:
    fun.plot_unscaled_eigenvectors(G)

if error:
    err = fun.error(G)
    print("error: ",err)
    print("error portion: ", err/max_inversions)

if plot_inversions:
    fun.plot_inversions(G)

if plot_eigenvectors:
    second, third = fun.get_eigenvectors(G)
    fun.plot_eigenvector(second, third)
