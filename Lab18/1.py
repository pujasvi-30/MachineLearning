import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

data = pd.DataFrame({
    'x1': [6, 6, 8, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 14],
    'x2': [5, 9, 6, 8, 10, 2, 5, 10, 13, 5, 8, 6, 11, 4, 8],
    'Label': ['Blue', 'Blue', 'Red', 'Red', 'Red', 'Blue', 'Red', 'Red', 'Blue',
              'Red', 'Red', 'Red', 'Blue', 'Blue', 'Blue']
})

X = data[['x1', 'x2']]
y = data['Label']
le = LabelEncoder()
y_enc = le.fit_transform(y)
rbf_model = SVC(kernel='rbf', gamma='scale', C=1)
poly_model = SVC(kernel='poly', degree=3, C=1)
rbf_model.fit(X, y_enc)
poly_model.fit(X, y_enc)

def plot_boundary(model, title):
    x_min, x_max = X['x1'].min() - 1, X['x1'].max() + 1
    y_min, y_max = X['x2'].min() - 1, X['x2'].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    sns.scatterplot(x=X['x1'], y=X['x2'], hue=y, s=120)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


plot_boundary(rbf_model, "RBF Kernel Decision Boundary")
plot_boundary(poly_model, "Polynomial Kernel Decision Boundary")

print("RBF Training Accuracy :", rbf_model.score(X, y_enc))
print("Polynomial Accuracy   :", poly_model.score(X, y_enc))



