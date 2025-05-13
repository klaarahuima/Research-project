import matplotlib.pyplot as plt
import numpy as np

# Assuming you have 4 sets of related variables with 5 trials each
half_of_second_a = np.array([0, 0.53,-0.088,0.054, 0.037,0.026])
half_of_second_f= np.array([0.00066,0.002,0.0023,0.0012,0.0006,0.0003])
half_of_second_ps = np.array([3.11, -0.24, 2.6, 2.54, 2.5,2.5])
half_of_second_off = np.array([33.39, -0.43, -0.014, 0.002, 0.0003,0])

rest_of_second_a = np.array([0.22, 0.16, 0.114, 0.081, 0.057, -0.04])
rest_of_second_f= np.array([0.022,0.011,0.0056, 0.0028, 0.0014, 0.0007])
rest_of_second_ps = np.array([-2, 1.08, 1.05, 4.18, 4.17, 1.03])
rest_of_second_off = np.array([0 for i in range(6)])

half_of_third_a = np.array([-0.15, -0.096, -0.07, -0.048, 0.034, 0.024])
half_of_third_f= np.array([0.011, 0.006, 0.0032, 0.0016, 0.0008, 0.0004])
half_of_third_ps = np.array([-0.5, -0.68, 2.4, 2.36, 2.35, 2.34])
half_of_third_off = np.array([0 for i in range(6)])

rest_of_third_a = np.array([0.23, 0.16, 0.116, 0.082, 0.058, 0.041])
rest_of_third_f= np.array([0.028, 0.014, 0.0071, 0.0036, 0.0018, -0.0009])
rest_of_third_ps = np.array([-2.54, -2.62, 0.48, 0.46, 3.6, 5.84])
rest_of_third_off = np.array([0 for i in range(6)])

# Create a figure and a 2x2 subplot grid
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
x = [50, 100, 200, 400, 800, 1600]
# Plot the data on each subplot
axs[0, 0].plot(x, half_of_third_a)
axs[0, 0].set_title('Half of Third')
axs[0, 0].set_xlabel('Trial')
axs[0, 0].set_ylabel('Amplitude')
axs[0, 0].grid(True)

axs[0, 1].plot(x, half_of_third_f)
axs[0, 1].set_title('Half of Third')
axs[0, 1].set_xlabel('Trial')
axs[0, 1].set_ylabel('Frequency')
axs[0, 1].grid(True)

axs[1, 0].plot(x, half_of_third_ps)
axs[1, 0].set_title('Half of Third')
axs[1, 0].set_xlabel('Trial')
axs[1, 0].set_ylabel('Phase shift')
axs[1, 0].grid(True)

axs[1, 1].plot(x, half_of_third_off)
axs[1, 1].set_title('Half of Third')
axs[1, 1].set_xlabel('Trial')
axs[1, 1].set_ylabel('Offset')
axs[1, 1].grid(True)

# Layout so plots do not overlap
fig.tight_layout()

plt.show()
