import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

#compute hypothesis
def hypothesis(X,theta):
    return np.dot(X,theta)

#compute cost function
def cost_function(Xtheta,y):
    y=y.to_numpy().reshape(-1,1)
    dif=np.subtract(Xtheta,y)
    cost=np.sum(np.power(dif,2))
    cost_func=cost/2
    return cost_func

#compute theta values
def find_theta(X,Xtheta,y,alpha,j):
    y = y.to_numpy().reshape(-1, 1)
    dif=np.subtract(Xtheta,y)
    s=np.sum(dif[:,0] * X[:,j])
    salpha = s * alpha
    return salpha

def main():
    california_housing = fetch_california_housing(as_frame=True)
    X = california_housing.data
    y = california_housing.target
    #df=pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    #X=df.drop(columns=['disease_score','disease_score_fluct'])
    scaler=StandardScaler()
    X=scaler.fit_transform(X)
    #y=df['disease_score'].values.reshape(-1,1)
    m=X.shape[0]
    X=np.c_[np.ones(m),X]
    theta=np.zeros((X.shape[1],1),dtype=float)
    alpha=0.00001
    threshold_theta=0.001
    threshold_cost=0.001
    prev_cost=float('inf')
    m=[]
    n=[]
    theta_prev=theta.copy()
    for i in range(2000):
        Xtheta = hypothesis(X, theta)
        cost=cost_function(Xtheta,y)
        print(f"iteration {i + 1} ; cost: {cost} ; theta: {theta.ravel()}")

        #to determine the convergence point
        m.append(cost)
        n.append(i+1)
        for j in range(len(theta)):
            theta[j][0] -= find_theta(X, Xtheta, y, alpha, j)
        theta_diff=np.max(np.abs(theta-theta_prev))
        cost_diff=abs(prev_cost-cost)
        if theta_diff < threshold_theta or cost_diff < threshold_cost:
            print("Converged at iteration:",i+1)
            break
        theta_prev=theta.copy()
        prev_cost=cost
    print("MSE",mean_squared_error(y,Xtheta))
    print("r2",r2_score(y,Xtheta))
    a_array = np.array(m)
    b_array = np.array(n)
    plt.plot(b_array,a_array)
    plt.xlabel('iteration')
    plt.ylabel('cost function')
    plt.xlim(0,2000)
    plt.show()

if __name__ == "__main__":
    main()