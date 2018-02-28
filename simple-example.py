"""
Starts by creating a target function and own data set. Evaluates
target function on each data input to classify output as +1 or -1.

Author: Alex Lim
"""

# Import packages
import numpy as np
import matplotlib.pyplot as plt

# Create own target function f
m = 0.5                                 # slope
b = 1                                   # y-intercept
x = np.linspace(-10, 10, 256)           # equally spaced values from -10 to 10
f = m*x + b                             # line equation (target function)

# Create own data set D of 20 random points
D = np.random.multivariate_normal([1, 1], np.diag([1, 1]), size=20)
print('Data set:\n', D)

'''
Get correct value of the output y_n by evaluating the target function on each x_n.
Modify the markers for each data point on the plot to be red for +1 and black for
-1 classification.
'''
y_out = np.array([])                     # create an empty numpy array
fig = plt.figure()                       # create an empty figure object
plt.plot(x, f, "--", figure=fig)         # plot target function line

# Loop through each data input and evaluate output and plot
for i, x in enumerate(D):
    if x[1] > m*x[0]+b:
        print(x, 'is above target function. Output is +1')
        plt.scatter(x[0], x[1], c='red', marker='X', figure=fig)
        y_out = np.append(y_out, 1)
    else:
        print(x, 'is below target function. Output is -1')
        y_out = np.append(y_out, -1)
        plt.scatter(x[0], x[1], c='black', marker='X', figure=fig)

# Print output results
print('Output results:\n', y_out)

# Plot characteristics
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Simple Perceptron Example')
plt.show()


