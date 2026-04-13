import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


    #Load data
def load_data():
    california_housing = fetch_california_housing(as_frame=True)
    X = california_housing.data
    y=california_housing.target
    # print(X.shape)
    # print(y.shape)
    return X, y

    #split train-test set
def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

#normalization of the data
def normalization(X_train, X_test):
#     max_X_tr = X_train.max(axis=0)
#     min_X_tr = X_train.min(axis=0)
    X_train_new=(X_train - X_train.min())/(X_train.max()-X_train.min())
    X_test_new=(X_test - X_train.min())/(X_train.max()-X_train.min())
    # print(X_train_new)
    # print(X_test_new)
    return X_train_new, X_test_new

#standardization of the data
# def standardization(X_train, X_test):
#     X_train_new = (X_train - X_train.mean()) / X_train.std()
#     X_test_new = (X_test - X_train.mean()) / X_train.std()
#     print("mean is", X_train_new.mean())
#     print("std dev is", X_train_new.std())
#     return X_train_new, X_test_new


    #create and train the linear regression model
def train_model(X_train_new, y_train):
    model = LinearRegression()
    model.fit(X_train_new, y_train)
    return model
    #make prediction of target values

def test(model,X_test_new,y_test):
    y_pred = model.predict(X_test_new)
    print(y_pred[:5])
    mse = mean_squared_error(y_test, y_pred)
    print("mean scored error", mse)
    r2=r2_score(y_test, y_pred)
    print("r2 score", r2)
    return mse, r2, y_pred

def main():
    X,y=load_data()
    X_train, X_test, y_train, y_test = split_data(X,y)
    #X_scaled_train, X_scaled_test=standardize(X_train, X_test)
    X_train_new, X_test_new = normalization(X_train, X_test)
    #X_train_new, X_test_new = standardization(X_train, X_test)
    model = train_model(X_train_new, y_train)
    mse, r2, y_pred= test(model,X_test_new,y_test)
    plt.figure(figsize=[10,5])
    plt.scatter(y_test,y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='r')
    plt.xlabel("actual values")
    plt.ylabel("predicted values")
    plt.show()


if __name__ == '__main__':
    main()
