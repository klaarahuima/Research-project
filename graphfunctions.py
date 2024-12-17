import scipy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def draw_graph(G): #with networkx
    nx.draw(G, with_labels=False, node_color="lightblue", edge_color="gray", node_size=4)
    plt.show()

def random_graph_edge_threshold(order = 500, threshold_angle = 30): #threshold angle in degrees
    n_points = order  # Number of points
    threshold_angle_rad = np.deg2rad(threshold_angle)  # Convert to radians
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
    return G, angles

def plot_random_graph(G, angles, order = 500, threshold = 30):
    positions = {i: (np.cos(angle), np.sin(angle)) for i, angle in enumerate(angles)}
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos=positions, with_labels=False, node_size=1, node_color="red", edge_color="none")
    plt.title(f"Random Graph on Unit Circle ({order} nodes, {threshold}° threshold)")
    plt.show()

def get_eigenvectors(nx_graph):
    A = nx.laplacian_matrix(nx_graph).astype(float)
    eigenvalues, eigenvectors = eigsh(A, k=3, which='SM')
    second_eigenvector = eigenvectors[:, 1]  # Eigenvector corresponding to 2nd eigenvalue
    third_eigenvector = eigenvectors[:, 2]  # Eigenvector corresponding to 3rd eigenvalue
    return second_eigenvector, third_eigenvector

def eigenvector_coordinates_scaled(eigenvector_tuple):
    second_eigenvector, third_eigenvector = eigenvector_tuple
    x = second_eigenvector[0]
    y = third_eigenvector[0]
    distance = np.sqrt(x ** 2 + y ** 2)
    scale_factor = 1 / distance
    scaled_x = []
    scaled_y = []
    for i in range(len(second_eigenvector)):
        x = second_eigenvector[i]
        y = third_eigenvector[i]
        scaled_x.append(x * scale_factor)
        scaled_y.append(y * scale_factor)
    return scaled_x, scaled_y

def eigenvector_coordinates(eigenvector_tuple):
    second_eigenvector, third_eigenvector = eigenvector_tuple
    array_x = []
    array_y = []
    for i in range(len(second_eigenvector)):
        x = second_eigenvector[i]
        y = third_eigenvector[i]
        array_x.append(x)
        array_y.append(y)
    return array_x, array_y

def plot_unscaled_coordinates(coordinates):
    x, y = coordinates
    plt.scatter(x, y, s = 1, color='red', label='Points')

    # Add labels and title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Scatter Plot of Points According to 2nd and 3rd Eigenvectors")
    plt.legend()
    # Show grid and the plot
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def plot_scaled_coordinates(coordinates):
    scaled_x, scaled_y = coordinates
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
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')  # Equal scaling for x and y axes
    plt.title("2nd and 3rd Eigenvectors on the Unit Circle")
    plt.legend()

    # Show the plot
    plt.show()

def plot_unscaled_eigenvectors(nx_graph):
    vectors = get_eigenvectors(nx_graph)
    plot_unscaled_coordinates(vectors)





