import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    s= 1 / (1 + np.exp(-x))
    return s

# get input values
x = np.linspace(-5, 5, 50)
y = sigmoid(x)

# Plot the values
plt.figure()
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("sigmoid(x)")
plt.title("Sigmoid Function")
plt.show()

