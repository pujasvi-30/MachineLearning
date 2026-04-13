from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


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
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, X_val, y_val, X_tr, y_tr

def standardize(X_tr,X_test, X_val, X_train):
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_tr)
    X_scaled_main_train=scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)
    X_scaled_val= scaler.transform(X_val)
    return X_scaled_train, X_scaled_val, X_scaled_test, X_scaled_main_train

    #create and train the linear regression model
def train_model(X_scaled_train, y_tr):
    model = LinearRegression()
    model.fit(X_scaled_train, y_tr)
    return model

    # make prediction of target values using val set
def test(model,X_scaled_val,y_val):
    print("Testing... using val set")
    y_pred = model.predict(X_scaled_val)
    print(y_pred[:5])
    mse = mean_squared_error(y_val, y_pred)
    print("mean scored error", mse)
    r2=r2_score(y_val, y_pred)
    print("r2 score", r2)
    return mse, r2

    #train the entire model using main train set
def train_main_model(X_scaled_main_train, y_train):
    model_main = LinearRegression()
    model_main.fit(X_scaled_main_train, y_train)
    return model_main

    #test the main model
def test_main(model_main,X_scaled_test,y_test):
    print("Testing... using entire train set")
    y_pred = model_main.predict(X_scaled_test)
    print(y_pred[:5])
    mse = mean_squared_error(y_test, y_pred)
    print("mean scored error", mse)
    r2=r2_score(y_test, y_pred)
    print("r2 score", r2)
    return mse, r2


def main():
    X,y=load_data()
    X_train, X_test, y_train, y_test, X_val, y_val, X_tr, y_tr  = split_data(X,y)
    #X_scaled_train, X_scaled_test=standardize(X_train, X_test)
    # X_train_new, X_test_new = normalization(X_train, X_test)
    X_scaled_train, X_scaled_val, X_scaled_test, X_scaled_main_train = standardize(X_tr, X_test, X_val, X_train)
    model = train_model(X_scaled_train, y_tr)
    test(model, X_scaled_val, y_val)
    model_main = train_main_model(X_scaled_main_train, y_train)
    test_main(model_main, X_scaled_test, y_test)


if __name__ == '__main__':
    main()
