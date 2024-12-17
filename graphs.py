import scipy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

"""
n = 1000  # Number of nodes
p = 0.01  # Probability of edge creation
G = nx.erdos_renyi_graph(n, p)"""

#create graph
G = nx.circulant_graph(100,[1,2,3,4,5,6,7,8,9])

# Step 2: Convert to a SciPy sparse adjacency matrix
A = nx.laplacian_matrix(G).astype(float)  # Returns a scipy.sparse.csr_matrix

# Step 3: Compute eigenvalues and eigenvectors
# Compute the largest 3 eigenvalues/vectors (k=3)
# `which="LM"` specifies the largest in magnitude
eigenvalues, eigenvectors = eigsh(A, k=3, which='SM')

# Step 4: Extract the 2nd and 3rd eigenvectors
second_eigenvector = eigenvectors[:, 1]  # Eigenvector corresponding to 2nd eigenvalue
third_eigenvector = eigenvectors[:, 2]   # Eigenvector corresponding to 3rd eigenvalue

#scaling to be on the unit circle
x = second_eigenvector[0]
y = third_eigenvector[0]
distance = np.sqrt(x**2 + y**2)
scale_factor = 1 / distance
scaled_x = []
scaled_y = []

for i in range(len(second_eigenvector)):
    x = second_eigenvector[i]
    y = third_eigenvector[i]
    scaled_x.append(x*scale_factor)
    scaled_y.append(y*scale_factor)

#plot unit circle
theta = np.linspace(0, 2 * np.pi, 100)  # 100 points on the unit circle
x_circle = np.cos(theta)
y_circle = np.sin(theta)
plt.figure(figsize=(6, 6))
plt.plot(x_circle, y_circle, label='Unit Circle', color='black', linestyle='--')

# Plot the points corresponding to the second and third eigenvectors
plt.scatter(scaled_x, scaled_y, color="blue")

# Add labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')  # Equal scaling for x and y axes
plt.title("2nd and 3rd Eigenvectors on the Unit Circle")
plt.legend()

# Show the plot
plt.show()