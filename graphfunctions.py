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

def get_eigenvectors(nx_graph):
    """
    Returns 2nd and 3rd Laplacian eigenvectors of a networkx graph
    """
    A = nx.laplacian_matrix(nx_graph).astype(float)
    eigenvalues, eigenvectors = eigsh(A, k=3, which='SM')
    second_eigenvector = eigenvectors[:, 1]  # Eigenvector corresponding to 2nd eigenvalue
    third_eigenvector = eigenvectors[:, 2]  # Eigenvector corresponding to 3rd eigenvalue
    return second_eigenvector, third_eigenvector

def eigenvector_radial_angles(eigenvector_tuple):
    """
    Args:
        eigenvector_tuple: 2 vectors, one for each eigenvector.
    Returns: A list of the angles of each point of the 2d-embedding in polar coordinates
    in the order the vertices were created.
    """
    second, third = eigenvector_tuple
    n = len(second)
    radial_angles = []
    for i in range(n):
        x = second[i]
        y = third[i]
        theta = np.arctan2(y, x)  # Compute angle in radians
        if theta < 0:  # Convert to [0, 2pi] range
            theta += 2 * np.pi
        radial_angles.append(theta)
    return radial_angles

def count_wrong_positions(radial_angles):
    """
    radial_angles: A list of all angular coordinates of the points in a graph's 2d-embedding.
    Returns: how many pairs of vertices have been inverted.
    The code might be faulty
    """
    count = 0
    n = len(radial_angles)
    for i in range(n):
        for j in range(i+1, n):
            if radial_angles[i] > radial_angles[j]:
                count += 1
    return count

def error(nx_graph):
    """
    Returns: how many inversions the 2d embedding of a networkx graph has.
    The code might be faulty
    """
    second, third = get_eigenvectors(nx_graph)
    rads = eigenvector_radial_angles((second, third))
    err = count_wrong_positions(rads)
    return err


def plot_unscaled_coordinates(coordinates):
    """
    Plots 2 vectors (with x and y coordinates, respectively) in the 2d-plane.
    """
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
    plt.scatter(0, 0, color="black", label="(0, 0)", zorder=1)
    plt.show()

def plot_unscaled_eigenvectors(nx_graph):
    """
    Directly plots the 2d-embedding og a networkx graph.
    """
    vectors = get_eigenvectors(nx_graph)
    plot_unscaled_coordinates(vectors)

def angles_to_dictionary(angles, order):
    """
    Takes two arguments: a list of angles (angular coordinates of vertices) and the order of the graph.
    Returns: A dictionary where every angle is assigned a position (ordering).
    """
    unsorted = [abs(x) for x in angles]
    vertex_angles = sorted(unsorted)
    keys = [i for i in range(order)]
    dictionary = dict(zip(keys, vertex_angles))
    return dictionary

def uniform_angles(n):
    """
    Generates n vertices uniformly around the unit circle
    """
    unsorted = np.random.uniform(0, 2 * np.pi, n)
    return angles_to_dictionary(unsorted, n)

def non_uniform_angles(n):
    """
    Generates n vertices non-uniformly around the unit circle.
    There are approximately twice as many vertices in the first quadrant than in the other quadrants.
    """
    raw = np.random.uniform(-np.pi/2, 2*np.pi, n)
    return angles_to_dictionary(raw, n)

def vertex_histogram(dictionary):
    """
    Plots a histogram visualising how many vertices are in each quadrant of the circular graph.
    """
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
    """
    Args:
        order of graph
        threshold angle
        p: probability that two vertices within the threshold distance are connected
        q: probability that two vertices not within the threshold distance are connected
    Returns: a networkx graph
    """
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
    """
    Plots graph with vertices on the unit circle.
    """

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
    """
    This was copied from one of the Applied Graph Theory workbooks.
    """
    average_degree = sum(degree for node, degree in nx_graph.degree()) / nx_graph.number_of_nodes()
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

    return average_degree









