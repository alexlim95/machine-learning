"""
This code shows an example of plotting a line as well as 20 random data points.
Will be used in the final perceptron learning algorithm example.

Author: Alex Lim
"""

# Import packages
import numpy as np
import matplotlib.pyplot as plt

# Obtain a initial line equation
m = 2
b = 1
x_line = np.linspace(-3, 3)     # x-axis points from -3 to 3
y = m*x_line + b                # line equation

fig = plt.figure()              # create a figure object

# Add 20 random sample points to the plot
x_data = np.random.multivariate_normal([1, 1], np.diag([1, 1]), size=20)
for i in range(x_data.shape[0]):
    plt.scatter(x_data[i, 0], x_data[i, 1], figure=fig, c='black', marker='X')

plt.plot(x_line, y, "--", figure=fig)   # creates a line plot
plt.xlabel('x-axis')                    # x-axis label
plt.ylabel('y-axis')                    # y-axis label
plt.title('Line Plot Example')          # plot label
plt.show()                              # show plot


