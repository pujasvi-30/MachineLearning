from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from ISLP import load_data
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree, DecisionTreeRegressor, DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


def load():
    data = load_data('Boston')
    print(data.shape)
    print(data.columns)
    X=data.drop('medv',axis=1)
    y=data['medv']
    print(X.shape)
    print(y.shape)
    return X,y

def split_data(X,y):
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=42)
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
    model=GradientBoostingRegressor(max_features=None, learning_rate=0.1, n_estimators=150, random_state=42, max_depth=3)
    model.fit(X_scaled_train, y_train)
    y_pred_final = model.predict(X_scaled_test)
    r2=r2_score(y_test, y_pred_final)
    mse_score=mean_squared_error(y_test, y_pred_final)
    # acc=accuracy_score(y_test, y_pred_final)
    print(r2)
    print(mse_score)
    # print(acc)
    print(model.estimators_.shape)
    for r in range(5):
        tree= model.estimators_[r][0]   #this shows 1st tree, [n_estimators],[0]- always 0 in case of regression as at each stage there is only one output
        plt_tree(tree, feature_names=X.columns)
    # print(tree.get_depth())    #prints actual depth of the tree

if __name__ == '__main__':
    main()


