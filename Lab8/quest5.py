import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder

from Lab8.quest4 import ordinal_encode


# load data
def load_data():
    df = pd.read_csv('breast_cancer.csv')
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    X = df.iloc[:,0:8].astype(str)
    # X = x.drop(columns=['diagnosis', 'id'])
    y = df.iloc[:,-1].astype(str)
    # print(df.isnull().sum())
    # print((df=='?').sum())
    # y=Y.map({'no-recurrence-events':0,'recurrence-events':1})  #using this will automatically convert empty data to NaN
    # print(y.unique(), "unique elements")
    # print(X.shape)
    # print(y.shape)
    # print(Y.unique)
    return X, y

# split train-test set
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_test=y_test.to_numpy().ravel()     #convert it to ndarray of  1D shape
    y_train = y_train.to_numpy().ravel()
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    return X_train, X_test, y_train, y_test

# standardize the data
def standardize(X_train, X_test, y_train, y_test):
    # scaler = StandardScaler()
    # X_scaled_train = scaler.fit_transform(X_train)
    # X_scaled_test = scaler.transform(X_test)
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    ordinal_encoder.fit(X_train)
    # onehot_encoder = OneHotEncoder(categories='auto', handle_unknown='ignore', sparse_output=False)
    # onehot_encoder.fit(X_train)
    X_encoded_train = ordinal_encoder.transform(X_train)
    X_encoded_test = ordinal_encoder.transform(X_test)
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_labeled_train = label_encoder.transform(y_train)
    y_labeled_test = label_encoder.transform(y_test)
    return X_encoded_train, X_encoded_test, y_labeled_train, y_labeled_test

# create and train the linear regression model
def train_model(X_encoded_train, y_labeled_train):
    model = LogisticRegression()
    model.fit(X_encoded_train, y_labeled_train)
    print("Coefficients: \n", model.coef_)
    print("Intercept: \n", model.intercept_)
    return model

# make prediction of target values
def test(model, X_encoded_test, y_labeled_test):
    print("actual values:", y_labeled_test[:5])
    y_pred = model.predict(X_encoded_test)
    print("predicted y values:", y_pred[:5])
    accuracy = accuracy_score(y_labeled_test, y_pred)
    print("accuracy score", accuracy)
    return accuracy


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_encoded_train, X_encoded_test, y_labeled_train, y_labeled_test = standardize(X_train, X_test, y_train, y_test)
    model = train_model(X_encoded_train, y_labeled_train)
    test(model, X_encoded_test, y_labeled_test)


if __name__ == '__main__':
    main()
