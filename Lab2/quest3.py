import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


#import os
#os.environ["http_proxy"] = "http://255hsbd018@ibab.ac.in:pujasviibab@proxy.ibab.ac.in:3128"
#os.environ["https_proxy"] ="http://255hsbd018@ibab.ac.in:pujasviibab@proxy.ibab.ac.in:3128"


    #Load data
def load_data():
    california_housing = fetch_california_housing(as_frame=True)
    #print(california_housing.DESCR)
    #print(california_housing.frame.head())
    #print(california_housing.target.head())
    #print(california_housing.data.head())
    #print(california_housing.frame.info())
    X = california_housing.data
    y=california_housing.target
    print(X.shape)
    print(y.shape)
    return X, y

    #split train-test set
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    return X_train, X_test, y_train, y_test

    #standardize the data
def standardize(X_train, X_test):
    scaler=StandardScaler()
    X_scaled_train=scaler.fit_transform(X_train)
    X_scaled_test=scaler.transform(X_test)
    return X_scaled_train, X_scaled_test

    #create and train the linear regression model
def train_model(X_scaled_train, y_train):
    model = LinearRegression()
    model.fit(X_scaled_train, y_train)
    return model
    #make prediction of target values

def test(model,X_test,y_test):
    y_pred = model.predict(X_test)
    print(y_pred[:5])
    mse = mean_squared_error(y_test, y_pred)
    print("mean scored error", mse)
    r2=r2_score(y_test, y_pred)
    print("r2 score", r2)
    return mse, r2

def main():
    X,y=load_data()
    X_train, X_test, y_train, y_test = split_data(X,y)
    X_scaled_train, X_scaled_test=standardize(X_train, X_test)
    model = train_model(X_scaled_train, y_train)
    test(model,X_scaled_test,y_test)




if __name__ == '__main__':
    main()


