import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler

# load data
def load_data():
    df = pd.read_csv('sonar data.csv')
    X = df.iloc[:,0:59]
    Y = df.iloc[:,60]
    y=Y.map({'R':0,'M':1})  #using this will automatically convert empty data to NaN
    print(y.unique(), "unique elements")
    print(X.shape)
    print(y.shape)
    return X, y

def k_fold_test(X, y, k=10):
    # length = len(X)
    fold = KFold(n_splits=k, shuffle=True, random_state=42)
    mse=[]
    r2=[]
    folds=[]
    for i, (train_index, test_index) in enumerate(fold.split(X)):
        # X_train, X_test = X[train_index], X[test_index]   #works for ndarray
        # y_train, y_test = y[train_index], y[test_index]
        X_train = X.iloc[train_index] #works for data frames
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        scaler= StandardScaler()
        scaler.fit_transform(X_train)
        scaler.transform(X_test)

        model= train_model(X_train, y_train)

        y_pred = model.predict(X_test)
        MSE= mean_squared_error(y_test, y_pred)
        R2= r2_score(y_test, y_pred)
        mse.append(MSE)
        r2.append(R2)
        folds.append(i)
        print(y_pred, "y values")
        print("MSE: Fold",i+1, mse)
        print("R2: Fold",i+1, r2)
        print("=Avg MSE(performance of model) ", np.mean(mse),"=standard deviation", np.std(mse))
    plt.plot(folds, r2)
    plt.xlabel('Number of Folds')
    plt.ylabel('R2')
    plt.title('R2 vs. Fold')
    plt.show()

def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def main():
    X, y = load_data()
    return k_fold_test(X, y)


if __name__ == '__main__':
    main()
