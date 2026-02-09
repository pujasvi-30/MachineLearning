import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


#X=[[1,2],[3,4]]
#y=[[5],[11]]

def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
    return X_train, X_test, y_train, y_test

def standardize(X_train, X_test):
    m = X_train.shape[0]
    X_train = np.c_[np.ones(m), X_train]
    n=X_test.shape[0]
    X_test = np.c_[np.ones(n), X_test]
    scaler=StandardScaler()
    X_scaled_train=scaler.fit_transform(X_train)
    X_scaled_test=scaler.transform(X_test)
    return X_scaled_train, X_scaled_test

def normal_equation(X, y):

    #to calculate X transpose
    row=len(X)
    col=len(X[0])

    XT=[[0 for _ in range(row)] for _ in range(col)]
    for i in range(row):
        for j in range(col):
            XT[j][i]=X[i][j]


    #to calculate X transpose X value
    rows=len(XT)
    cols=len(XT[0])
    XTX=[[0 for _ in range(col)]for _ in range(rows)]
    for i in range(rows):
        for j in range(col):
            for k in range(cols):
                XTX[i][j]+=XT[i][k]*X[k][j]
    # XTX=np.array(XTX, dtype=float)
    XTXI=np.linalg.inv(XTX)  #calculate the inverse value of XTransposeX

    ro=len(XTXI)
    co=len(XTXI[0])
    cols = len(XT[0])
    XTXIXT=[[0 for _ in range(cols)]for _ in range(ro)]
    for i in range(ro):
        for j in range(cols):
            for k in range(co):
                XTXIXT[i][j]+=XTXI[i][k]*XT[k][j]

    r=len(XTXIXT)
    c=len(XTXIXT[0])
    coly=len(y[0])

    theta=[[0 for _ in range(coly)] for _ in range(r)]
    for i in range(r):
        for j in range(coly):
            for k in range(c):
                theta[i][j]+=XTXIXT[i][k]*y[k][j]
    return theta


def main():
    df=pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    x=df.iloc[:,0:5]
    X=x.to_numpy()          #to convert dataframe series into array
    y = df["disease_score"].to_numpy().reshape(-1, 1)
    X_train, X_test, y_train, y_test=split_data(X,y)
    X_scaled_train, X_scaled_test= standardize(X_train, X_test)
    theta=normal_equation(X_scaled_train, y_train)
    print("theta values computed through normal equation", theta)
    theta=np.array(theta)
    y_pred=np.dot(theta, X_scaled_test)
    print("MSE", mean_squared_error(y_test, y_pred))
    print("r2", r2_score(y_test, y_pred))

    # scaler=StandardScaler()
    # X=scaler.fit_transform(X)
    # m = X.shape[0]
    # X = np.c_[np.ones(m), X]

if __name__ == '__main__':
    main()
