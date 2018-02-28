"""
This example shows the perceptron learning algorithm (update rule) applied to
a logical AND. The desired output is already known. Weights are initialized
to 0. The goal is to observe the converged values of the weights.
"""

import numpy as np

'''
AND gate desired output:

(0,0) --> 0 (y = -1)
(0,1) --> 0 (y = -1)
(1,0) --> 0 (y = -1)
(1,1) --> 1 (y = +1)
'''
y = [-1, -1, -1, 1]

'''
Input variables (2 dimensional) - 2-bit AND.
Contains an extra variable Xo which is 1.
'''
x = ([1, 0, 0],
     [1, 0, 1],
     [1, 1, 0],
     [1, 1, 1])

'''
Initialize weights to 3.
'''
w = np.zeros(3)

'''
Iterate through all data points.
'''
max_iterations = 10
for t in range(max_iterations):
    for i in range(4):
        # check if misclassified or not --> update weights
        if (np.dot(x[i], w)*y[i]) <= 0:
            print('Found a misclassified data point:', x[i])
            w[0] = w[0] + x[i][0]*y[i]      # update w0
            w[1] = w[1] + x[i][1]*y[i]      # update w1
            w[2] = w[2] + x[i][2]*y[i]      # update w2
        else:
            print("")

''' Final weight vector '''
print(w)




