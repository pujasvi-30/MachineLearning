import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
from sklearn.preprocessing import OneHotEncoder


def loadData():
    # data=load_diabetes()
    data=load_iris()
    X=pd.DataFrame(data.data, columns=data.feature_names)
    y=pd.Series(data.target)
    y.fillna(y.mean(), inplace=True)
    print(X.isna().sum())
    return X,y

def k_fold(X,y,k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold=[]
    acc=[]
    # mse=[]
    # r2=[]
    for i, (train_index, test_index) in enumerate (kf.split(X)):
        X_train=X.iloc[train_index]
        X_test=X.iloc[test_index]
        y_train=y.iloc[train_index]
        y_test=y.iloc[test_index]



        model= train_model(X_train, y_train)

        y_pred=model.predict(X_test)

        accuracy=accuracy_score(y_test, y_pred)
        acc.append(accuracy)
        # MSE=mean_squared_error(y_test, y_pred)
        # r=r2_score(y_test, y_pred)
        # mse.append(MSE)
        # r2.append(r)
        fold.append(i)
        print(y_pred[:5])
        # print("mse:", MSE)
        # print("r2:", r)
        print("accuracy:", i+1, accuracy)

def train_model(X, y):
    model = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1, max_depth=10, min_samples_split=5,
                                    min_samples_leaf=2, max_features='sqrt')
    # model = RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=-1, max_depth=10, min_samples_split=5,min_samples_leaf = 2, max_features = 'sqrt')

    # model=Ridge(alpha=0.01)

    model.fit(X, y)
    return model

# def splitData(X,Y):
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
#     return X_train, X_test, Y_train, Y_test
# #
# def scale(X_train, X_test):
#     # encoder = OneHotEncoder(categories='auto',handle_unknown='ignore')
#     # X_train = encoder.fit_transform(X_train)
#     # X_test = encoder.transform(X_test)
#     scaler = StandardScaler()
#     X_train=scaler.fit_transform(X_train)
#     X_test=scaler.transform(X_test)
#     return X_train, X_test

def main():
    X,y=loadData()
    return k_fold(X,y)
    # X_train, X_test, Y_train, Y_test = splitData(X,y)
    # X_train, X_test= scale(X_train, X_test)


    # modelrf = RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=-1, max_depth=10, min_samples_split=5, min_samples_leaf=2, max_features='sqrt')
    # cv_fold = cross_val_score(modelrf, X_train, Y_train, cv=5)
    # modelrf.fit(X_train, Y_train)
    # print("cv_fold_mean_rf:",cv_fold.mean())
    # print("cv_fold_std_rf:", cv_fold.std())
    # y_pred = modelrf.predict(X_test)
    # print("y_pred_rf", y_pred[:5])
    # mse=mean_squared_error(Y_test, y_pred)
    # r2=r2_score(Y_test, y_pred)
    # print("mse", mse)
    # print("r2", r2)

    # modelrfc= RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1, max_depth=10, min_samples_split=5, min_samples_leaf=2, max_features='sqrt')
    # cv_fold = cross_val_score(modelrfc, X_train, Y_train, cv=5)
    # modelrfc.fit(X_train, Y_train)
    # print("cv_fold_mean_rfc:",cv_fold.mean())
    # print("cv_fold_std_rfc:", cv_fold.std())
    # y_pred = modelrfc.predict(X_test)
    # print("y_pred_rf", y_pred[:5])
    # accuracy=accuracy_score(y_pred, Y_test)
    # print("accuracy", accuracy)

    # tree=modelrf.estimators_[0]
    # plt.figure(figsize=(12, 8))
    # plot_tree(tree, feature_names=X.columns, filled=True)
    # plt.show()

if __name__ == '__main__':
    main()