import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#X=[[1,2],[3,4]]
#y=[[5],[11]]

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
    X=x.to_numpy()    #to convert dataframe series into array
    y = df["disease_score"].to_numpy().reshape(-1, 1)
    scaler=StandardScaler()
    X=scaler.fit_transform(X)
    m = X.shape[0]
    X = np.c_[np.ones(m), X]
    theta=normal_equation(X, y)
    print("theta values computed through normal equation", theta)


if __name__ == '__main__':
    main()
