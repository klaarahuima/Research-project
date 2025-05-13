import networkx as nx
import numpy as np
import graphfunctions as fun
import matplotlib.pyplot as plt

"""
Used to observe the number of inversions in large graphs, specifically the relationship between
proportion of inverted vertices vs. graph order.
Will comment more soon.
"""

orders = [500,750,1000,1500,2000]

with open("simulation_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Order", "Error Proportion"])  # Column headers

def run_simulations(orders,p = 1, q =0, num_simulations=3):
    t = 72
    avg_inversions = []
    avg_error_proportions = []
    with open("simulation_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for n in orders:
            errors = 0
            runs = 0
            total_error_proportions = 0
            #total_pairs = n * (n - 1) / 2
            for i in range(num_simulations):
                print(f'simulating order {n} run n. {i+1}')
                G = fun.random_graph_unit_circle(n, t, p, q)
                fun.add_eigenvectors(G, 0, 1)
                if nx.is_connected(G):
                    error = fun.error(G)
                    errors += error

                    runs += 1
                    writer.writerow([n, error])
                    # Calculate error proportion (inversions / total possible inversions)
                    num_pairs = n * (n - 1) // 2  # Number of possible pairs of vertices
                    error_proportion = error / num_pairs if num_pairs > 0 else 0
                    total_error_proportions += error_proportion

                # Calculate the average values for this graph order
            avg_inversions.append(errors / runs)
            avg_error_proportions.append(total_error_proportions / runs)

    return avg_inversions, avg_error_proportions

avg_inversions, avg_error_proportions = run_simulations(orders)

# Save each iteration's result
# Run the simulations and get the average proportion of inversions
# Assuming run_simulations(orders) returns two lists


# Plotting the results
plt.figure(figsize=(8, 6))

# Plot average inversions
plt.plot(orders, avg_error_proportions, marker='o', color='r')

# Corrected title method
plt.title('Average Error Proportion (Inversions / Total Pairs)')

# Corrected axis labels
plt.xlabel('Order')
plt.ylabel('Average Error Proportion')

plt.grid(True)  # Add gridlines
plt.tight_layout()
plt.show()
