import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import circulants as cry
import graphfunctions as fun

"""
Files used to plot sinusoidal parameters of the eigenvector equations with respect to graph order.
I used this code to determine that the parameters followed a power-law relationship with respect to graph order.

The code will calculate the parameters of 4 sinusoidal fits: 1st quadrant / 2nd - 4th quadrant of
2nd laplacian eigenvector / 3rd laplacian eigenvector.

Running the file as is will print a dictionary with the parameters for the graph sizes you define below in the orders array.
"""


orders = [200,300,400,500,600,700,800,900,1000] # orders of graphs of which you want to plot their sinusoidal parameters
# below: initializing empty dictionary for storing parameters
# s1: 1st quadrant of second eigenvector
# s2: 2nd-4th quadrant of 2nd eigenvector
# t1: 1st quadrant of third eigenvector
# t2: 2nd-4th quadrant of 3rd eigenvector
# stored variables: amplitude, frequency, phi, offsets
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

# body of code: generating graphs of all defined orders
for n in orders:
    n_cry = n//5*2 # number of vertices in first quadrant
    # You can choose what kind of graphs to generate, uncomment which model you want to use
    # I compared random graphs and ideal circulant graphs (as defined in circulants.py, and called by cry.generate_graph(n))
    # G = cry.generate_graph(n)
    G = fun.random_graph_unit_circle(n, threshold, p, q)
    fun.add_eigenvectors(G, 0, 1)
    vectors = cry.split_vectors(G, n_cry) # getting a 4-tuple of the vectors we are studying
    for i in range(len(vectors)): # iterate over the 4 vectors
        y_data = vectors[i] # eigenvector data
        x_data = [i for i in range(len(y_data))] # creating list of indices

        # Define the sinusoidal function
        if i == 0 or i == 1: # 2nd eigenvector, 1st quadrant (i==0) or 2nd-4th quadrant (i==1)
            # in this case I assume its a cosine function
            def sinusoidal(x, A, freq, phi, offset):
                return A * np.cos(2*np.pi * freq * x + phi) + offset
        else:
            # otherwise its the 3rd eigenvector, and I assume its a sine function
            def sinusoidal(x, A, freq, phi, offset):
                return A * np.sin(2*np.pi * freq * x + phi) + offset
        # getting correct label based on which vector it is
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
        initial_guesses = [0.05, 0.0005, 1, 0] # guessing the 4 defining parameters
        # If the fit looks completely wrong, you might want to tweak these and see what works
        print(f'Fitting order {n} and vector {label}')

        # Fit the sinusoidal function to the data
        # using curve_fit from scipy.optimize
        params, covariance = curve_fit(sinusoidal, x_data, y_data, p0=initial_guesses, maxfev=1000000)
        A_fit, freq_fit, phi_fit, offset_fit = params # extracting parameters
        mod_p= (phi_fit % np.pi + np.pi) % np.pi # normalizing phi

        # adding the fitted parameters to the dictionary in which we store all of them
        v['amps'][label].append(A_fit)
        v['freqs'][label].append(freq_fit)
        v['phis'][label].append(mod_p)
        v['offs'][label].append(offset_fit)


        #print(f"Fitted Parameters for vector_{i}:\nAmplitude: {A_fit}\nFrequency: {freq_fit}\nPhase Shift: {phi_fit}\nOffset: {offset_fit}")

        # Here we are getting points from the fitted sinusoidal curve
        x_fit = np.linspace(min(x_data), max(x_data), 500)  # Dense x values for a smooth curve
        y_fit = sinusoidal(x_fit, A_fit, freq_fit, phi_fit, offset_fit) # Y-values

        # Here we are plotting the empirical data and fitted curve
        plt.figure(figsize=(8, 6))
        plt.scatter(x_data, y_data, label="Eigenvector coordinates", color="red", alpha=0.8, s = 5)
        plt.plot(x_fit,y_fit, label="Fitted sinusoidal curve", color="blue", alpha=0.8, linewidth=1.3)
        plt.title(title)
        plt.legend()
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.show()
print(v)


"""
Here onwards I wanted to study how the parameters change wrt the order. Uncommenting this part will create a plot
that visualizes each parameter ( 4 parameters for each vector = 16 in total)

# defining 4 x 4 grid
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
plt.show()
"""