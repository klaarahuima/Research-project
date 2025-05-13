import graphfunctions as fun
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.optimize import curve_fit

"""
Functions used to generate ideal circulant graphs with exactly twice as many vertices in 1st quadrant (not random).
Will comment further soon.
"""

def generate_graph(n):
    G = nx.Graph()
    # Add the first 40 vertices and edges for the circulant graph
    num_circulant = n//5*2
    k = 2  # Number of closest neighbors on each side
    G.add_nodes_from([i for i in range(n)])
    G.add_edge(0, 1)
    G.add_edge(0,2)
    G.add_edge(1,3)
    G.add_edge(1, 2)
    for i in range(2,num_circulant-2):
        G.add_edge(i, (i + 1))  # Add edge to the right
        G.add_edge(i, (i - 1))  # Add edge to the left
        G.add_edge(i, (i - 2))  # Add edge to the left
        G.add_edge(i, (i + 2))  # Add edge to the left
    G.add_edge(num_circulant-2,num_circulant-1)

    for i in range(num_circulant,n-1):
        G.add_edge(i, (i + 1))  # Add edge to the next vertex
        G.add_edge(i, (i - 1)) # Add edge to the previous vertex
    G.add_edge(n-1,0)
    fun.add_eigenvectors(G, 0, 1)

    return G

#fun.plot_eigenvector(G.graph['second_lap'], G.graph['third_lap'])

#fun.plot_eigenvector(G.graph['second_lap'][2:num_circulant-2], G.graph['third_lap'][2:num_circulant-2])

#fun.plot_eigenvector(G.graph['second_lap'][num_circulant:], G.graph['third_lap'][num_circulant:])

def split_vectors(G, num_circulant):
    s = G.graph['second_lap'][2:num_circulant-2]
    p = G.graph['second_lap'][num_circulant:]
    r = G.graph['third_lap'][2:num_circulant-2]
    q = G.graph['third_lap'][num_circulant:]
    return (s, p ,r, q)

#print(radians)
#print(multipliers)
"""

def plot_sinusoidal_by_angle(angles):

    # Wrap angles into [0, 2π] range
    wrapped_angles = np.mod(angles, 2 * np.pi)

    # Calculate the sinusoidal values (cosine of the angles)
    sinusoidal_values = np.cos(wrapped_angles)

    # Plot the sinusoidal function
    plt.figure(figsize=(10, 5))
    plt.plot(wrapped_angles, sinusoidal_values, marker='o', linestyle='-', label="Sinusoidal Function")
    plt.title("Sinusoidal Function vs. Angle")
    plt.xlabel("Angle (radians)")
    plt.ylabel("Value")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(np.pi, color='red', linestyle='--', label=r"$\pi$")
    plt.axvline(2 * np.pi, color='blue', linestyle='--', label=r"$2\pi$")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_sinusoidal_by_angle(radians)

def plot_angles(angles):
    # Wrap angles into [0, 2π] range
    wrapped_angles = np.mod(angles, 2 * np.pi)

    # Create indices for x-axis
    indices = np.arange(len(angles))

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(indices, wrapped_angles, marker='o', linestyle='-', label="Wrapped Angles")
    plt.axhline(2 * np.pi, color='red', linestyle='--', label='$2\pi$ Boundary')
    plt.axhline(0, color='blue', linestyle='--', label='0 Boundary')
    plt.title("Angles Wrapped into [0, 2π] Range")
    plt.xlabel("Index")
    plt.ylabel("Angle (radians)")
    plt.legend()
    plt.grid(True)
    plt.show()

import numpy as np
from scipy.optimize import curve_fit
import pylab as plt

N = 40 # number of data points
t = np.linspace(0, 2*np.pi, N)
data = G.graph['second_lap'][:40]

def sine_function(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

# Initial guess for the parameters [A, B, C, D]
initial_guess = [0.02, 1, 0, 0]

x = np.linspace(0, 40, 40)
y = G.graph['second_lap'][:40]
# Perform the curve fitting
params, covariance = curve_fit(sine_function, x, y, p0=initial_guess)

# Extract the fitted parameters
A_fit, B_fit, C_fit, D_fit = params

print(f"Fitted parameters: A={A_fit}, B={B_fit}, C={C_fit}, D={D_fit}")
# Generate y values using the fitted parameters
y_fit = sine_function(x, A_fit, B_fit, C_fit, D_fit)

# Plot the original data and the fitted curve
plt.scatter(x, y, label='Sample Data')
plt.plot(x, y_fit, label='Fitted Curve', color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sine Curve Fitting')
plt.legend()
plt.show()
"""