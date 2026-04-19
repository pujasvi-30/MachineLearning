import numpy as np
x1 = [3, 6]
x2 = [10, 10]
K=x1[0]**2 * x2[0]**2 + 2*x1[0]*x2[0]*x1[1]*x2[1] + x1[1]**2 * x2[1]**2
print("Kernel:",K)
n=np.dot(x1,x2)
print("Normally:",n**2)