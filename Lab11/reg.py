import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


class DecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = None
    def fit(self, X, y):
        data = np.column_stack((X, y))
        self.tree = self._build_tree(data, 0)
    def _build_tree(self, data, depth):
        X, y = data[:, :-1], data[:, -1]
        if len(y) <= self.min_samples_split or depth >= self.max_depth:
            return np.mean(y)
        feature, split = self._best_split(data)
        if feature is None:
            return np.mean(y)
        left = data[data[:, feature] < split]
        right = data[data[:, feature] >= split]
        return {
            "feature": feature,
            "split": split,
            "left": self._build_tree(left, depth + 1),
            "right": self._build_tree(right, depth + 1)
        }

    def _best_split(self, data):
        X = data[:, :-1]
        best_error = float("inf")
        best_feature, best_split = None, None
        for feature in range(X.shape[1]):
            for split in np.unique(X[:, feature]):
                left = data[X[:, feature] < split]
                right = data[X[:, feature] >= split]
                if len(left) == 0 or len(right) == 0:
                    continue
                error = self._mse(left[:, -1]) + self._mse(right[:, -1])
                if error < best_error:
                    best_error = error
                    best_feature = feature
                    best_split = split
        return best_feature, best_split

    def _mse(self, y):
        return np.sum((y - np.mean(y)) ** 2)

    def _predict_one(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        if x[tree["feature"]] < tree["split"]:
            return self._predict_one(x, tree["left"])
        else:
            return self._predict_one(x, tree["right"])
    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])



data = pd.read_csv("Boston.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)
