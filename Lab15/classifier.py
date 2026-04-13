from sklearn.ensemble import GradientBoostingClassifier
from ISLP import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def load():
    data = load_data('Weekly')
    print(data.shape)
    print(data.columns)
    X=data.drop('Direction',axis=1)
    y=data['Direction']
    y=y.map({'Up':1,'Down':0})
    # print(y.fillna(y.mean(),inplace=True))
    print(X.shape)
    print(y)
    return X,y

def split_data(X,y):
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=42)
    # y_test = y_test.to_numpy().ravel()
    # y_train = y_train.to_numpy().ravel()
    return X_train, X_test, y_train, y_test

def standardize(X_train, X_test):
    scaler = StandardScaler()
    X_scaled_train= scaler.fit_transform(X_train)
    X_scaled_test= scaler.transform(X_test)
    return X_scaled_train, X_scaled_test

def plt_tree(tree,feature_names):
    plt.figure(figsize=(12, 8))
    plot_tree(tree, feature_names=feature_names)
    plt.show()

def main():
    X, y= load()
    X_train, X_test, y_train, y_test= split_data(X, y)
    X_scaled_train, X_scaled_test= standardize(X_train, X_test)
    model=GradientBoostingClassifier(max_features=None, learning_rate=0.1, n_estimators=150, random_state=42, max_depth=3)
    model.fit(X_scaled_train, y_train)
    y_pred_final = model.predict(X_scaled_test)
    acc=accuracy_score(y_test, y_pred_final)
    print(acc)
    print(model.estimators_.shape)
    for r in range(5):
        tree= model.estimators_[r][0]   #this shows 1st tree, [n_estimators],[0]- always 0 in case of regression as at each stage there is only one output
        plt_tree(tree, feature_names=X.columns)
    # print(tree.get_depth())    #prints actual depth of the tree

if __name__ == '__main__':
    main()


