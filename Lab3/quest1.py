import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

    #load data
def load_data():
    df=pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    print(df.head())
    print(df.describe())
    print(df.shape)
    print(df.info)
    X=df.iloc[:,0:6]
    y=df["disease_score"]
    print(X.shape)
    print(y.shape)
    return X,y

    #split train-test set
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
    print("Coefficients: \n", model.coef_)
    print("Intercept: \n", model.intercept_)
    return model

    #make prediction of target values
def test(model,X_test,y_test):
    print("actual values:", y_test.values[:5])
    y_pred = model.predict(X_test)
    print("predicted y values:", y_pred[:5])
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
