import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler

# load data
def load_data():
    df = pd.read_csv('wisconsin.csv')
    X = df.iloc[:,2:31]
    # X = x.drop(columns=['diagnosis', 'id'])
    Y = df["diagnosis"]
    y=Y.map({'B':0,'M':1})  #using this will automatically convert empty data to NaN
    # print(y.unique(), "unique elements")
    # print(X.shape)
    # print(y.shape)
    return X, y

# split train-test set
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_test=y_test.to_numpy().ravel()     #convert it to ndarray of  1D shape
    y_train = y_train.to_numpy().ravel()
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    return X_train, X_test, y_train, y_test

# standardize the data
def standardize(X_train, X_test):
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)
    return X_scaled_train, X_scaled_test

# create and train the logistic regression model
def train_model(X_scaled_train, y_train):
    # model = LogisticRegression(penalty='l2', solver='liblinear')
    model = LogisticRegression(penalty='l1', solver='liblinear')
    # model=Ridge(alpha=0.01)    #for linear regression regularization
    model.fit(X_scaled_train, y_train)
    print("Coefficients: \n", model.coef_)
    print("Intercept: \n", model.intercept_)
    return model

# make prediction of target values
def test(model, X_scaled_test, y_test):
    print("actual values:", y_test[:5])
    y_pred = model.predict(X_scaled_test)
    print("predicted y values:", y_pred[:5])
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy score", accuracy)
    return accuracy


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_scaled_train, X_scaled_test = standardize(X_train, X_test)
    model = train_model(X_scaled_train, y_train)
    test(model, X_scaled_test, y_test)


if __name__ == '__main__':
    main()
