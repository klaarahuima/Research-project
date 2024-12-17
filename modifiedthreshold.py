import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import graphfunctions as fun


order = 100
threshold_degree = 15
p = 0.8
q = 0.001

neighbourhood = order/360 * threshold_degree * 0.8
#print("neighbourhood size: ", neighbourhood)

threshold = np.deg2rad(threshold_degree)


def angles_to_dictionary(angles):
    unsorted = [abs(x) for x in angles]
    vertex_angles = sorted(unsorted)
    keys = [i for i in range(order)]
    dictionary = dict(zip(keys, vertex_angles))
    return dictionary

def uniform_angles(order):
    unsorted = np.random.uniform(0, 2 * np.pi, order)
    return angles_to_dictionary(unsorted)

def non_uniform_angles(order):
    raw = np.random.uniform(-np.pi/2, 2*np.pi, order)
    return angles_to_dictionary(raw)

def degree_histogram(dictionary):
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

    # Plotting the histogram
    labels = ['Quadrant 1', 'Quadrant 2', 'Quadrant 3', 'Quadrant 4']
    plt.bar(labels, quadrant_counts, color='skyblue')

    # Adding titles and labels
    plt.title('Distribution of Vertex Angles in Different Quadrants')
    plt.xlabel('Quadrant')
    plt.ylabel('Number of Vertices')
    plt.show()

def random_graph_unit_circle(order, threshold, p, q):
    # generate angles uniformly for the vertices

    dictionary =  non_uniform_angles(order)
    degree_histogram(dictionary)

    G = nx.Graph()
    for i in dictionary.keys():
        G.add_node(i, angle=dictionary[i])  # Add nodes with their angle as an attribute

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

    return G, dictionary


def plot_graph_unit_circle(G, dictionary, title="Random Graph on Unit Circle"):
    """
    Plot the graph on the unit circle.

    Args:
        G (networkx.Graph): The graph to plot.
        angles (np.ndarray): Array of vertex angles on the unit circle.
        title (str): Title for the plot.
    """
    # Map vertices to 2D coordinates on the unit circle
    positions = {i: (np.cos(angle), np.sin(angle)) for i, angle in dictionary.items()}

    # Plot the graph
    plt.figure(figsize=(8, 8))
    nx.draw(
        G,
        pos=positions,
        with_labels=False,
        node_size=3,
        node_color="blue",
        edge_color="lightgray"
    )
    plt.title(title)
    plt.axis("equal")
    plt.show()


G, angles = random_graph_unit_circle(order, threshold, p, q)
print("size: ",G.size())
fun.draw_graph(G)
# Plot the graph
plot_graph_unit_circle(G, angles, title=f"Random Graph (n={order}, p={p}, q={q}, threshold={np.rad2deg(threshold):.1f}°)")

fun.plot_unscaled_eigenvectors(G)