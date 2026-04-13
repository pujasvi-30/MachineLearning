import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor, BaggingClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.datasets import load_diabetes
from sklearn.tree import plot_tree


def load():
    # data=load_iris()
    data=load_diabetes()
    X=pd.DataFrame(data.data, columns=data.feature_names)
    y=pd.Series(data.target)
    y.fillna(y.mean(), inplace=True)
    return X,y

def k_fold(X, y, k=10, shuffle=True):
    fold=KFold(n_splits=k, shuffle=shuffle, random_state=42)
    folds=[]
    #acc=[]
    mse=[]
    r2=[]
    for i, (train_index, test_index) in enumerate (fold.split(X)):
        X_train=X.iloc[train_index]
        y_train=y.iloc[train_index]
        X_test=X.iloc[test_index]
        y_test=y.iloc[test_index]

        scaler=StandardScaler()
        X_train=scaler.fit_transform(X_train)
        X_test=scaler.transform(X_test)

        model= train_model(X_train, y_train)

        model.predict(X_test)
        y_pred=model.predict(X_test)
        # accuracy=accuracy_score(y_test, y_pred)
        MSE=mean_squared_error(y_test, y_pred)
        r=r2_score(y_test, y_pred)
        mse.append(MSE)
        r2.append(r)
        folds.append(i+1)
        # acc.append(accuracy)
        # print("accracy scor:", i+1, accuracy)
        print("MSE:",i+1, MSE)
        print("r2:",i+1, r)
    # plt.plot(folds, acc)
    # plt.xlabel("k")
    # plt.ylabel("accuracy_score")
    # plt.show()


def train_model(X_train, y_train):
    model=BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=10, max_features=None, min_samples_leaf=2, min_samples_split=8), n_estimators=100, random_state=42, max_samples=0.8)
    #model = BaggingClassifier(
        #estimator=DecisionTreeClassifier(max_depth=5, max_features='sqrt', min_samples_leaf=2, min_samples_split=5),
        #n_estimators=100, random_state=42, max_samples=0.8)
    #model=RandomForestRegressor(random_state=42, n_estimators=200, max_depth=8, min_samples_split=5, min_samples_leaf=2)
    #model=DecisionTreeRegressor(random_state=42, max_depth=5, min_samples_split=2, min_samples_leaf=2, max_features=None)
    model.fit(X_train, y_train)
    return model

def plot(X,y):
    # tree = DecisionTreeClassifier(    max_depth=5, random_state=42)
    tree= DecisionTreeRegressor(max_depth=5, random_state=42)
    tree.fit(X, y)
    plt.figure(figsize=(12, 8))
    plot_tree(tree, feature_names=X.columns, filled=True)
    plt.show()

def main():
    X,y=load()
    k_fold(X, y)
    plot(X,y)

if __name__=='__main__':
    main()