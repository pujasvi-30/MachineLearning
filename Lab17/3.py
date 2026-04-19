from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.array([
    [0.4, -0.7], [-1.5, -1.0], [-1.4, -0.9], [-1.3, -1.2],
    [-1.1, -0.2], [-1.2, -0.4], [-0.5, 1.2], [-1.5, 2.1],
    [1.0, 1.0], [1.3, 0.8], [1.2, 0.5], [0.2, -2.0],
    [0.5, -2.4], [0.2, -2.3], [0.0, -2.7], [1.3, 2.1]
])

y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])


def plot_training_data_with_decision_boundary(kernel, ax=None, support_vectors=True):
    # Train model
    clf = svm.SVC(kernel=kernel, gamma=2).fit(X, y)

    # Create plot if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    x_min, x_max, y_min, y_max = -3, 3, -3, 3
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Decision boundary (color background)
    DecisionBoundaryDisplay.from_estimator(
        clf, X,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
        ax=ax
    )

    # Contour lines (margin + boundary)
    DecisionBoundaryDisplay.from_estimator(
        clf, X,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors="k",
        linestyles=["--", "-", "--"],
        ax=ax
    )

    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolors="k")

    # Support vectors
    if support_vectors:
        ax.scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=150,
            facecolors="none",
            edgecolors="k"
        )

    # Legend
    ax.legend(*scatter.legend_elements(), title="Classes")

    ax.set_title(f"{kernel} kernel SVM")


# Run for different kernels
kernels = ["linear", "rbf", "poly"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, k in zip(axes, kernels):
    plot_training_data_with_decision_boundary(k, ax=ax)

plt.tight_layout()
plt.show()