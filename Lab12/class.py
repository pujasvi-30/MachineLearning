import numpy as np
import math
from collections import Counter

def entropy(y):
    counts = Counter(y)
    total = len(y)
    ent = 0
    for count in counts.values():
        prob = count / total
        ent += -prob * math.log2(prob)
    return ent

def information_gain(X_column, y, threshold):
    parent_entropy = entropy(y)
    left_mask = X_column <= threshold
    right_mask = X_column > threshold
    if sum(left_mask) == 0 or sum(right_mask) == 0:
        return 0
    n = len(y)
    n_left = sum(left_mask)
    n_right = sum(right_mask)
    e_left = entropy(y[left_mask])
    e_right = entropy(y[right_mask])
    child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
    ig = parent_entropy - child_entropy
    return ig

def best_split(X, y, num_features):
    features = np.random.choice(X.shape[1], num_features, replace=False)
    best_gain = -1
    split_idx, split_thresh = None, None
    for feature in features:
        X_column = X[:, feature]
        thresholds = np.unique(X_column)
        for threshold in thresholds:
            gain = information_gain(X_column, y, threshold)
            if gain > best_gain:
                best_gain = gain
                split_idx = feature
                split_thresh = threshold
    return split_idx, split_thresh

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class RandomTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=10, num_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.num_features = num_features
        self.root = None

    def fit(self, X, y):
        if self.num_features is None:
            self.num_features = int(np.sqrt(X.shape[1]))  # random subset

        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping conditions
        if (depth >= self.max_depth or
                n_labels == 1 or
                n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # find best split
        feat_idx, threshold = best_split(X, y, self.num_features)

        if feat_idx is None:
            return Node(value=self._most_common_label(y))

        # split data
        left_mask = X[:, feat_idx] <= threshold
        right_mask = X[:, feat_idx] > threshold

        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        return Node(feat_idx, threshold, left, right)

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    data = load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42)
    clf = RandomTreeClassifier(max_depth=10)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))