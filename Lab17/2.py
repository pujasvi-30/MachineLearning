import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


data = [
    [6, 5, 'Blue'], [6, 9, 'Blue'], [8, 6, 'Red'], [8, 8, 'Red'], [8, 10, 'Red'],
    [9, 2, 'Blue'], [9, 5, 'Red'], [10, 10, 'Red'], [10, 13, 'Blue'],
    [11, 5, 'Red'], [11, 8, 'Red'], [12, 6, 'Red'], [12, 11, 'Blue'],
    [13, 4, 'Blue'], [14, 8, 'Blue']
]

df = pd.DataFrame(data, columns=['x1', 'x2', 'label'])

# Convert labels to numeric
df['label'] = df['label'].map({'Blue': 0, 'Red': 1})

X = df[['x1', 'x2']].values
y = df['label'].values



# RBF Kernel
rbf_model = SVC(kernel='rbf', gamma=0.5)
rbf_model.fit(X, y)
rbf_pred = rbf_model.predict(X)

# Polynomial Kernel
poly_model = SVC(kernel='poly', degree=3)
poly_model.fit(X, y)
poly_pred = poly_model.predict(X)



def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} RESULTS")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)
    print("Confusion Matrix:\n", cm)



evaluate_model("RBF Kernel", y, rbf_pred)
evaluate_model("Polynomial Kernel", y, poly_pred)



def plot_decision_boundary(model, title):
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=100)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()



plot_decision_boundary(rbf_model, "RBF Kernel Decision Boundary")
plot_decision_boundary(poly_model, "Polynomial Kernel Decision Boundary")