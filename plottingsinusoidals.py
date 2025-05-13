import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import circulants as cry
import graphfunctions as fun

"""
Files used to plot sinusoidals of the different parameters of the eigenvector equations.
Will comment more soon.

"""


# Example empirical data
#x = [i for i in range(500)]
orders = [1000]
v = {
    'amps': {'s1':[], 's2':[], 't1':[], 't2':[]},
    'freqs': {'s1':[], 's2':[], 't1':[], 't2':[]},
    'phis': {'s1':[], 's2':[], 't1':[], 't2':[]},
    'offs': {'s1':[], 's2':[], 't1':[], 't2':[]}
}
threshold_degree = 72 #max distance to 'near-by' vertices in degrees
p = 1 #probability of connecting two vertices within the threshold distance
q = 0 #probability of connecting two vertices not withing the threshold distance
threshold = np.deg2rad(threshold_degree)

for n in orders:
    n_cry = n//5*2
    #G = cry.generate_graph(n)
    G = fun.random_graph_unit_circle(n, threshold, p, q)
    fun.add_eigenvectors(G, 0, 1)
    vectors = cry.split_vectors(G, n_cry)
    for i in range(len(vectors)):
        y_data = vectors[i]
        x_data = [i for i in range(len(y_data))]

        # Define the sinusoidal function
        if i == 0 or i == 1:
            def sinusoidal(x, A, freq, phi, offset):
                return A * np.cos(2*np.pi * freq * x + phi) + offset
        else:
            def sinusoidal(x, A, freq, phi, offset):
                return A * np.sin(2*np.pi * freq * x + phi) + offset
        if i == 0:
            label = 's1'
            title = 'Second eigenvector, Q1'
        elif i == 1:
            label = 's2'
            title = 'Second eigenvector, Q2-Q4'
        elif i == 2:
            label = 't1'
            title = 'Third eigenvector, Q1'
        else:
            label = 't2'
            title = 'Third eigenvector, Q2-Q4'
        initial_guesses = [0.05, 0.0005, 1, 0]
        print(f'Fitting order {n} and vector {label}')

        # Fit the sinusoidal function to the data
        params, covariance = curve_fit(sinusoidal, x_data, y_data, p0=initial_guesses, maxfev=1000000)
        A_fit, freq_fit, phi_fit, offset_fit = params
        mod_p= (phi_fit % np.pi + np.pi) % np.pi


        v['amps'][label].append(A_fit)
        v['freqs'][label].append(freq_fit)
        v['phis'][label].append(mod_p)
        v['offs'][label].append(offset_fit)


        #print(f"Fitted Parameters for vector_{i}:\nAmplitude: {A_fit}\nFrequency: {freq_fit}\nPhase Shift: {phi_fit}\nOffset: {offset_fit}")
        x_fit = np.linspace(min(x_data), max(x_data), 500)  # Dense x values for a smooth curve
        y_fit = sinusoidal(x_fit, A_fit, freq_fit, phi_fit, offset_fit)

        # Plot empirical data and fitted curve
        plt.figure(figsize=(8, 6))
        plt.scatter(x_data, y_data, label="Eigenvector coordinates", color="red", alpha=0.8, s = 5)
        plt.plot(x_fit,y_fit, label="Fitted sinusoidal curve", color="blue", alpha=0.8, linewidth=1.3)
        plt.title(title)
        plt.legend()
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.show()
print(v)
"""print(v)
categories = ["amps", "freqs", "phis", "offs"]
labels = ["s1", "s2", "t1", "t2"]

# Create a 4x4 subplot layout
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
fig.suptitle("Sinusoidal Fit Parameters Across Orders", fontsize=16)

# Iterate over subcategories and labels to plot each individually
for row, category in enumerate(categories):
    for col, label in enumerate(labels):
        ax = axes[row, col]  # Get subplot position

        ax.plot(orders, v[category][label], marker="o", linestyle="-", color="b")

        ax.set_title(f"{category.capitalize()} - {label}")
        ax.set_xlabel("Order")
        ax.set_ylabel(category.capitalize())
        ax.grid(True)

# Adjust layout for readability
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()"""