
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def load_data():
    df = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    X = df.drop(columns=['disease_score', 'disease_score_fluct']).values
    y = df['disease_score'].values  #.values converts it into numpy array
    return X,y

def k_fold_test(X, y, k=10):
    # length = len(X)
    fold = KFold(n_splits=k, shuffle=True, random_state=42)
    mse=[]
    r2=[]
    folds=[]
    for i, (train_index, test_index) in enumerate(fold.split(X)):
        X_train, X_test = X[train_index], X[test_index]  #this works for ndarray only
        y_train, y_test = y[train_index], y[test_index]

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
        print("MSE: Fold",i+1, mse)
        print("R2: Fold",i+1, r2)
        print(np.mean(mse), "=Avg MSE(performance of model) ", np.std(mse), "=standard deviation")
    plt.plot(folds, r2)
    plt.xlabel('Number of Folds')
    plt.ylabel('R2')
    plt.title('R2 vs. Fold')
    plt.show()

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def main():
    X, y = load_data()
    return k_fold_test(X, y)

if __name__ == '__main__':
    main()

