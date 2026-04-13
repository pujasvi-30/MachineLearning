import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def k_fold(X, y, k=10):
    length= len(X)
    foldsize= length // k
    positions= np.arange(length)
    np.random.shuffle(positions)
    X= X[positions]
    y= y[positions]
    mse=[]
    r2=[]
    fold=[]
    for i in range(k):
        start = i * foldsize
        end = start + foldsize
        X_te= X[start:end]
        y_test= y[start:end]
        X_train = np.concatenate((X[:start], X[end:]),axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_te)

        model=train_model(X_train, y_train)
        y_pred = model.predict(X_val)
        MSE= mean_squared_error(y_test, y_pred)
        R2= r2_score(y_test, y_pred)
        mse.append(MSE)
        r2.append(R2)
        fold.append(i+1)
        print(" MSE | Fold",i+1, mse)
        print("R2 | Fold",i+1, r2)
        print(np.mean(mse), "=Avg MSE(performance of model) ", np.std(mse), "=standard deviation")
    plt.plot(fold,r2)
    plt.xlabel('fold')
    plt.ylabel('R2')
    plt.title('R2 vs. Fold')
    plt.show()

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def main():
    df=pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    X = df.drop(columns=['disease_score', 'disease_score_fluct']).values
    y = df['disease_score'].values
    return k_fold(X, y)

if __name__ == "__main__":
    main()




