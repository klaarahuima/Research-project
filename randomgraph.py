import scipy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

# Parameters
n_points = 500  # Number of points
threshold_angle_deg = 20  # Threshold in degrees
threshold_angle_rad = np.deg2rad(threshold_angle_deg)  # Convert to radians

# Step 1: Generate random angles on the unit circle
angles = np.random.uniform(0, 2 * np.pi, n_points)

# Step 2: Create the graph
G = nx.Graph()
# Add nodes with angles as attributes
for i, angle in enumerate(angles):
    G.add_node(i, angle=angle)

# Step 3: Connect points if their angular distance is within the threshold
for i in range(n_points):
    for j in range(i + 1, n_points):
        angle_diff = np.abs(angles[i] - angles[j])
        # Adjust for circular distance
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
        if angle_diff <= threshold_angle_rad:
            G.add_edge(i, j)

#plot 2nd and 3rd eigenvectors
A = nx.laplacian_matrix(G).astype(float)
eigenvalues, eigenvectors = eigsh(A, k=3, which='SM')
second_eigenvector = eigenvectors[:, 1]  # Eigenvector corresponding to 2nd eigenvalue
third_eigenvector = eigenvectors[:, 2]   # Eigenvector corresponding to 3rd eigenvalue


#plot unit circle
theta = np.linspace(0, 2 * np.pi, 100)  # 100 points on the unit circle
x_circle = np.cos(theta)
y_circle = np.sin(theta)
plt.figure(figsize=(6, 6))
plt.plot(x_circle, y_circle, label='Unit Circle', color='black', linestyle='--')

# Plot the points corresponding to the second and third eigenvectors
plt.scatter(scaled_x, scaled_y, s = 1,color="red")

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

# Step 4: Visualize the graph
# Get positions of nodes based on their angles
positions = {i: (np.cos(angle), np.sin(angle)) for i, angle in enumerate(angles)}

plt.figure(figsize=(8, 8))
nx.draw(
    G,
    pos=positions,
    with_labels=False,
    node_size=1,
    node_color="red",
    edge_color="none"
)
plt.title(f"Random Graph on Unit Circle ({n_points} nodes, {threshold_angle_deg}° threshold)")
plt.show()

#nx.draw(G, with_labels=False, node_color="lightblue", edge_color="gray", node_size=3)
