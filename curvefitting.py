
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

"""
File used to fit sinusoidal curves to eigenvectors.
Will comment more thoroughly soon.
"""

# Given data
y_data = np.array([0.07532658, 0.06530192, 0.05495845, 0.04434666, 0.03351835, 0.0225264, 0.01142447, 0.00026676, -0.01089225, -0.02199808, -0.03299651, -0.04383384, -0.05445717, -0.06481462, -0.07485563, -0.08453118, -0.09379402, -0.10259894, -0.11090294, -0.11866548, -0.12584867, -0.13241744, -0.13833971, -0.14358657, -0.1481324, -0.15195501, -0.15503574, -0.15735954, -0.15891508, -0.15969475, -0.15969475, -0.15891508, -0.15735954, -0.15503574, -0.15195501, -0.1481324, -0.14358657, -0.13833971, -0.13241744, -0.12584867, -0.11866548, -0.11090294, -0.10259894, -0.09379402, -0.08453118, -0.07485563, -0.06481462, -0.05445717, -0.04383384, -0.03299651, -0.02199808, -0.01089225, 0.00026676, 0.01142447, 0.0225264, 0.03351835, 0.04434666, 0.05495845, 0.06530192, 0.07532658])



# Generate x values (assuming x values are evenly spaced from 0 to 1)
x_data = np.linspace(0, 1, len(y_data))

# Define the sinusoidal function
def sinusoidal(x, A, f, phi, offset):
    return A * np.cos(2 * np.pi * f * x + phi) + offset

# Initial guess for the parameters
p0 = [0.16, 0.05, np.pi, 0]
# Perform curve fitting
popt, pcov = curve_fit(sinusoidal, x_data, y_data, p0=p0, method='dogbox')
# Print the estimated parameters
print("Estimated parameters:")
print("Amplitude (A):", popt[0])
print("Frequency (f):", popt[1])
print("Phase shift (phi):", popt[2])
print("Vertical shift (offset):", popt[3])

# Generate x values for plotting the estimated function
x_plot = np.linspace(0, 1, 1000)

# Calculate the estimated function values
y_plot = sinusoidal(x_plot, *popt)

# Plot the data and the estimated function
plt.plot(x_data, y_data, 'bo', label='Data')
plt.plot(x_plot, y_plot, 'r-', label='Estimated function')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data and Estimated Sinusoidal Function')
plt.show()
