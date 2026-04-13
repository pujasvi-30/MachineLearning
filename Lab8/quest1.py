import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

#to split and standardize the given data
def split_data(df):
    split = int(0.7 * len(df))
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    X_train = train_df.drop(columns=['disease_score', 'disease_score_fluct'])
    X_test = test_df.drop(columns=['disease_score', 'disease_score_fluct'])
    y_train = train_df['disease_score'].values.reshape(-1, 1)
    y_test = test_df['disease_score'].values.reshape(-1, 1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    m = X_train.shape[0]
    X_train = np.c_[np.ones(m), X_train]
    n = X_test.shape[0]
    X_test = np.c_[np.ones(n), X_test]
    return X_train, X_test, y_train, y_test

#calculate the hypothesis function
def hypothesis(X,theta):
    return np.dot(X,theta)

#to calculate the cost function
def cost_function(Xtheta,y,lamb, theta):
    dif=np.subtract(Xtheta,y)
    cost=np.sum(np.power(dif,2))
    cost_func=cost/2
    # l2norm=lamb*np.sum(theta[1:]**2)
    l1norm=lamb*np.sum(np.abs(theta))
    # final_cost=cost_func+l2norm
    final_cost=cost_func+l1norm
    return final_cost

#to compute theta values
def find_theta(X,Xtheta,y,alpha,j,lamb,theta):
    dif=np.subtract(Xtheta,y)
    s=np.sum(dif[:,0] * X[:,j])
    if j!=0:
        # s+=lamb * theta[j,0]
        s+=lamb*np.sign(theta[j,0])
    salpha = s * alpha
    return salpha

def main():
    df=pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    X_train, X_test, y_train, y_test=split_data(df)
    theta=np.zeros((X_train.shape[1],1),dtype=float)
    alpha=0.001
    lamb=0.01
    threshold_theta=0.001
    threshold_cost=0.001
    prev_cost=float('inf')
    m=[]
    n=[]
    theta_prev=theta.copy()
    for i in range(1000):
        Xtheta = hypothesis(X_train, theta)
        cost=cost_function(Xtheta,y_train, lamb, theta)
        print(f"iteration {i + 1} ; cost: {cost} ; theta: {theta.ravel()}")

        #to determine the point of convergence
        m.append(cost)
        n.append(i+1)
        for j in range(len(theta)):
            theta[j][0] -= find_theta(X_train, Xtheta, y_train, alpha, j, lamb, theta)
        theta_diff=np.max(np.abs(theta-theta_prev))
        cost_diff=abs(prev_cost-cost)
        if theta_diff < threshold_theta or cost_diff < threshold_cost:
            print("Convergence at iteration:",i+1)
            break
        theta_prev=theta.copy()
        prev_cost=cost
    print("MSE",mean_squared_error(y_train,Xtheta))
    print("r2",r2_score(y_train,Xtheta))
    a_array = np.array(m)
    b_array = np.array(n)
    plt.plot(b_array,a_array)
    plt.xlabel('iteration')
    plt.ylabel('cost function')
    plt.xlim(0,2000)
    plt.show()

if __name__ == "__main__":
    main()
