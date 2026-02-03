a = len(x)
dot = 0
for i in range(a):
    dot += x[i] * y[i]
print("Dot product:",dot)

#using numpy

import numpy as np
x = np.array([2,1,2])
y = np.array([1,2,2])

dot = np.dot(x,y)
print("Dot product using numpy:",dot)

"""multiplying the corresponding elements of two vectors and 
adding them up to compute a scalar value.
"""
