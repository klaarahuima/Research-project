import networkx as nx
import numpy as np
import graphfunctions as fun
import duplicatingfunctions as dup
import knpaths as path
from graphfunctions import get_adj_eigenvalues, get_adjacency_eigenvectors
import circulants as cry

"Define parameters of graph"
order = 2400
threshold_degree = 30 #max distance to 'near-by' vertices in degrees
p = 1 #probability of connecting two vertices within the threshold distance
q = 0 #probability of connecting two vertices not withing the threshold distance
likelihood = 2

threshold = np.deg2rad(threshold_degree)
max_inversions = order*(order-1)/2


#G= fun.random_graph_unit_circle(order, threshold, p, q)
G = cry.generate_graph(order)
#fun.add_eigenvectors(G, 0, 1)

#s_tuple = (G.graph['second_lap'], G.graph['third_lap'])

#rads, distances = fun.eigenvector_polar_coordinates(s_tuple)
#fun.plot_distances(distances)

#G_1 = nx.circulant_graph(1000,[i for i in range(1,45)])
# #G_3 = nx.circulant_graph(1000,[i for i in range(1, 90)])


#G = nx.hypercube_graph(4)
#G = (nx.circulant_graph(20, [1,2,3]))
#G= dup.create_graph(400,100)
#G = nx.path_graph(100)
#G = path.k_path_graph(12,2)

#G = fun.nearest_neighbours(10000, 1500)
"""
second, third = get_adjacency_eigenvectors(G)
result = 0
for i in range(len(second)):
    result += second[i]*third[i]

print(result)
"""

"""
graphs = [G_1, G_2, G_3]
for graph in graphs:
    fun.add_eigenvectors(graph, 1, 1)
    print('new graph')
    print(graph.graph['a_eigenvalues'])
    print(graph.graph['l_eigenvalues'])
    """


"Choose what to display"
graph_networkx = False #networkx default visualisation
graph_circle = False #vertices and edges on the unit circle

adjacency_embedding = False #embedding of 2nd and 3rd eigenvectors
laplacian_embedding = True

adj_eigenvectors = False
lap_eigenvectors = True

split_vectors = True

print('Adjacency eigenvalues: ', G.graph['a_eigenvalues'])
print('Laplacian eigenvalues: ', G.graph['l_eigenvalues'])
print('Average degree: ', 2* G.size()/G.order())

adj_error = False #kendall distance of ordering of vertices
lap_error = False

vertex_distribution = False #histogram of how many vertices are in each quadrant
degree_distribution = False #degree histogram
plot_inversions = False

if graph_networkx:
    fun.draw_graph(G)

if graph_circle:
    fun.plot_graph_unit_circle(G, title=f"Random Graph")

if adjacency_embedding:
    if nx.is_connected(G):
        fun.plot_unscaled_eigenvectors(G, "adj")
    else:
        print('not connected')

if laplacian_embedding:
    if nx.is_connected:
        fun.plot_unscaled_eigenvectors(G, "lap")

if adj_eigenvectors:
    fun.plot_eigenvector(G.graph['second_adj'], G.graph['third_adj'])

if lap_eigenvectors:
    fun.plot_eigenvector(G.graph['second_lap'], G.graph['third_lap'])

if split_vectors:
    fun.plot_eigenvector(G.graph['second_lap'][:2 * order // 5], G.graph['third_lap'][:2 * order // 5])
    fun.plot_eigenvector(G.graph['second_lap'][2 * order // 5:], G.graph['third_lap'][2 * order // 5:])

if adj_error:
    err = fun.error(G, "adj")
    print("adjacency error: ",err)
    print("adjacency error portion: ", err/max_inversions)

if lap_error:
    err = fun.error(G, "lap")
    print("laplacian error: ",err)
    print("laplacian error portion: ", err/max_inversions)

if vertex_distribution:
    fun.vertex_histogram(G.graph['dictionary'])

if degree_distribution:
    avg = fun.plot_degree_distribution(G)
    print("average degree: ", avg)



#avg_neighbourhood = order/360 * threshold_degree * 2 * p
#print("average number of neighbours within threshold degree: ", avg_neighbourhood)


