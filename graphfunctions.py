import scipy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from holoviews.plotting.bokeh.styles import font_size
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
    second, third = eigenvector_tuple
    array_x = []
    array_y = []
    for x, y in zip(second, third):
        array_x.append(x)
        array_y.append(y)
    return array_x, array_y

def eigenvector_radial_coordinates(eigenvector_tuple):
    second, third = eigenvector_tuple
    radial_angles = []
    radial_distances = []
    for x, y in zip(second, third):
        r = np.sqrt(x ** 2 + y ** 2)  # Compute radial distance
        theta = np.arctan2(y, x)  # Compute angle in radians
        radial_angles.append(theta)
        radial_distances.append(r)
    return radial_distances, radial_angles

def count_wrong_positions(radial_angles):
    count = 0
    n = len(radial_angles)
    for i in range(n):
        for j in range(i+1, n):
            if radial_angles[i] < radial_angles[j]:
                count += 1
    return count

def kendall_distance(radial_angles):
    sorted_indices = np.argsort(radial_angles)  # Indices of the angles in sorted order
    inverse_permutation = np.argsort(sorted_indices)  # Original -> New position

    # Step 2: Count inversions using a double loop
    inversions = 0
    n = len(inverse_permutation)
    for i in range(n):
        for j in range(i + 1, n):
            if inverse_permutation[i] > inverse_permutation[j]:
                inversions += 1

    return inversions


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

def angles_to_dictionary(angles, order):
    unsorted = [abs(x) for x in angles]
    vertex_angles = sorted(unsorted)
    keys = [i for i in range(order)]
    dictionary = dict(zip(keys, vertex_angles))
    return dictionary

def uniform_angles(order):
    unsorted = np.random.uniform(0, 2 * np.pi, order)
    return angles_to_dictionary(unsorted, order)

def non_uniform_angles(order):
    raw = np.random.uniform(-np.pi/2, 2*np.pi, order)
    return angles_to_dictionary(raw, order)

def vertex_histogram(dictionary):
    quadrant_counts = [0, 0, 0, 0]  # Counts for each quadrant
    #counting quadrant elements
    for angle in dictionary.values():
        if 0 <= angle < np.pi / 2:
            quadrant_counts[0] += 1
        elif np.pi / 2 <= angle < np.pi:
            quadrant_counts[1] += 1
        elif np.pi <= angle < 3 * np.pi / 2:
            quadrant_counts[2] += 1
        else:
            quadrant_counts[3] += 1
    labels = ['Quadrant 1', 'Quadrant 2', 'Quadrant 3', 'Quadrant 4']
    plt.bar(labels, quadrant_counts, color='skyblue')
    plt.title('Distribution of Vertices in Different Quadrants')
    plt.xlabel('Quadrant')
    plt.ylabel('Number of Vertices')
    plt.show()

def random_graph_unit_circle(order, threshold, p, q):
    # generate angles uniformly for the vertices
    angle_dict = non_uniform_angles(order)

    G = nx.Graph(dictionary = angle_dict)
    for i in angle_dict.keys():
        G.add_node(i, angle=angle_dict[i])  # Add nodes with their angle as an attribute

    # Step 3: Connect edges based on the distance and probabilities
    for i in range(order):
        for j in range(i + 1, order):
            # Calculate the angular distance between nodes i and j
            abs_diff = abs(G.nodes[i]["angle"] - G.nodes[j]["angle"])
            angle_diff = min(abs_diff, 2 * np.pi - abs_diff)  # Circular distance

            # Decide connection probability
            prob = p if angle_diff < threshold else q

            # Add edge with the computed probability
            if np.random.rand() < prob:
                G.add_edge(i, j)

    return G

def plot_graph_unit_circle(nx_graph, title="Random Graph on Unit Circle"):

    # Map vertices to 2D coordinates on the unit circle
    positions = {i: (np.cos(angle), np.sin(angle)) for i, angle in nx_graph.graph['dictionary'].items()}

    # Plot the graph
    plt.figure(figsize=(8, 8))
    nx.draw(
        nx_graph,
        pos=positions,
        with_labels=False,
        node_size=3,
        node_color="blue",
        edge_color="lightgray"
    )
    plt.title(title)
    plt.axis("equal")
    plt.show()

def plot_degree_distribution(nx_graph):
    # this was copied from one of the applied graph theory workbooks
    degree_sequence = sorted((degree for node, degree in nx_graph.degree()), reverse=True)
    min_degree = min(degree_sequence)
    max_degree = max(degree_sequence)
    degree_range = max_degree - min_degree + 1
    degrees = range(min_degree, max_degree + 1)  # create a list of numbers from min degree through max degree

    deg_distribution = np.zeros(degree_range)  # initiate all counts equal to zero
    for v in nx_graph:
        d = nx_graph.degree(v)
        deg_distribution[d - min_degree] += 1  # increases the relevant counter by 1

    fig, ax = plt.subplots(figsize=(8, 6))  # Standard figure and axes creation
    ax.bar(degrees, deg_distribution, width=0.8, color='lightblue', edgecolor='none')  # Bar chart with better aesthetics
    ax.set_xticks(degrees)  # Set x tick marks to the degree values
    plt.xticks(fontsize=5)
    ax.set_xlabel("Degree", fontsize=12)  # Label for the x-axis
    ax.set_ylabel("Frequency", fontsize=12)  # Label for the y-axis
    ax.set_title("Degree Distribution", fontsize=14)  # Add a title to the plot
    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.show()









