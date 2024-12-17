import graphfunctions as fun

order = 200
threshold_angle= 35

#create random graph by sampling uniformly on unit circle and using a threshold angle to create edges
G, angles = fun.random_graph_edge_threshold(order, threshold_angle)

print(G)
#plot the graph on a unit circle
fun.plot_random_graph(G, angles, order, threshold_angle)

#get coordinates of 2nd and 3rd eigenvectors and plot them
eigvectors = fun.get_eigenvectors(G)
cords = fun.eigenvector_coordinates(eigvectors)
fun.plot_unscaled_coordinates(cords)

print(eigvectors)