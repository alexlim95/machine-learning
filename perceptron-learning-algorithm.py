"""
This code demonstrates the perceptron learning algorithm. An ideal target function
f and data set D were created in order to see how the algorithm works. This example
shows the perceptron in a two-dimensional Euclidean space (d=2). The output y_out was
found by evaluating the target function on each of the 20 randomly generated inputs.
The perceptron learning algorithm was applied to the data set in order to converge
to a final hypothesis g function and compared to the target function f.

The blue line is the target function f.
The green line is the converged hypothesis g decision boundary line.

The way the perceptron learning algorithm works is by iterating through each input point
and checking whether it is classified correctly. If a point is misclassified (i.e the sign
of the dot product of the weights w and the input vector x is opposite the output y), then
the algorithm updates each of the weights accordingly using the update rule w = w + y*x.

Author: Alex Lim
"""

# Import packages
import numpy as np
import matplotlib.pyplot as plt

# Create own target function f
m_tar = 0.5                                  # slope
b_tar = 1                                    # y-intercept
x_tar = np.linspace(-3, 3)                   # equally spaced values from -3 to 3
f_tar = m_tar*x_tar + b_tar                  # line equation (target function)

'''
Create own data set D of 20 random points. The mean determines where
the points are mostly likely to be generated. The covariance is the 
level to which two variables vary together (higher covariance results
in a more spread apart data points).
'''
D = np.random.multivariate_normal(mean=[0, 1], cov=np.diag([1, 1]), size=20)

# Include x0 input term
x0 = 1                                       # x0 = 1
D = np.insert(D, 0, x0, axis=1)              # insert x0 to first element of input vector
print('Data set:\n', D)                      # print out input data set

'''
Get correct value of the output y_n by evaluating the target function on each x_n.
Modify the markers for each data point on the plot to be red for +1 and black for
-1 classification.
'''
y_out = np.array([])                                                            # create an empty numpy array
fig = plt.figure()                                                              # create an empty figure object
plt.plot(x_tar, f_tar, "--", figure=fig, color='b', label="Target Function")    # plot target function line

# Loop through each data input and evaluate output and plot appropriate color (+1=red -1=black)
for i, x in enumerate(D):
    # Check if input is above the target function threshold (map to +1)
    if x[2] > m_tar*x[1]+b_tar:
        print(x, 'is above target function. Output is +1')
        plt.scatter(x[1], x[2], c='red', marker='o', figure=fig)    # plot point as a red X
        y_out = np.append(y_out, 1)                                 # append output to array
    # Input is below the target function threshold (map to -1)
    else:
        print(x, 'is below target function. Output is -1')
        y_out = np.append(y_out, -1)                                # append output to array
        plt.scatter(x[1], x[2], c='black', marker='o', figure=fig)  # plot point as a black X

# Print output results
print('Output results:\n', y_out)

# Initialize weight vector to all 1. w0 is the bias
w = np.array([1., 1., 1.])

# Continue to loop and update weights
iter_count = 0                                              # counter to see how many iterations take place
update_count = 0                                            # counter to see how many weight updates are done
error_found = False                                         # boolean variable to check if misclassified input is found
while True:
    for i, x in enumerate(D):                               # single loop through each input: j(index) x(input)
        if (np.dot(x, w)*y_out[i]) <= 0:                    # checks for misclassified input
            error_found = True
            print('Found a misclassified input:', x)
            # Apply the update rule to the weights
            w[0] = w[0] + x[0]*y_out[i]
            w[1] = w[1] + x[1]*y_out[i]
            w[2] = w[2]  +x[2]*y_out[i]
            update_count += 1                               # increment weight update counter
            print('New value of weights after update:', w)
        else:
            print('Input correctly classified:', x)
    iter_count += 1  # increment iteration count
    # Reset error_found boolean variable if misclassified input was found in iteration
    if error_found is True:
        error_found = False
    # if went through whole data set without finding a misclassified input --> break loop --> DONE
    elif error_found is False:
        break

# Number of iterations needed
print('Number of iterations needed:', iter_count)

# Number of weight updates needed
print('Number of weight updates needed:', update_count)

# Print the weights upon completion of iteration
print('Weight values upon completion:', w)

'''
Plotting the decision curve. The equation is: w1*x1 + w2*x2 + b = 0
where b is the bias w0.

We check two cases: 
    (1) x = 0 and y != 0
    (2) y = 0 and x != 0

Two equations: 
    (1) w1*x + b = 0
    (2) w2*y + b = 0

Bounds:
    (1) x = -b/w1
    (2) y = -b/w2

Build the line equation:
    f(z) = c*z + d

Two more unknowns:
    f(0) = c*0 + d = y
    f(x) = c*x + d = 0

Result:
    d = y
    c = -d/x = -y/x
    
In terms of weights:
    slope = -(w[0]/w[2])/(w[0]/w[1])
    intercept = -w[0]/w[2]
    y_line = slope*x + intercept
'''
slope = -(w[0]/w[2])/(w[0]/w[1])                                             # slope of line
intercept = -w[0]/w[2]                                                       # y-intercept
y_line = slope*x_tar + intercept                                             # line equation
plt.plot(x_tar, y_line, "--", figure=fig, color='g', label="Hypothesis")     # plot decision boundary

# Plot characteristics
plt.xlabel('X1 Axis')                            # x-axis label
plt.ylabel('X2 Axis')                            # y-axis label
plt.grid(True)                                   # turn on grid
plt.title('Perceptron Learning Algorithm')       # title
plt.legend()                                     # show legend
plt.show()                                       # show plot

