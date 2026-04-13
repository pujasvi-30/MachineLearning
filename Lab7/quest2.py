import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
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
    folds=[]
    acc=[]
    for i, (train_index, test_index) in enumerate(fold.split(X)):
        # X_train, X_test = X[train_index], X[test_index]
        # y_train, y_test = y[train_index], y[test_index]
        X_train = X.iloc[train_index] #works for data frames
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        X_train_new, X_test_new=normalization(X_train, X_test)
        # scaler= StandardScaler()
        # scaler.fit_transform(X_train)
        # scaler.transform(X_test)

        model= train_model(X_train_new, y_train)

        y_pred = model.predict(X_test_new)

        accu= accuracy_score(y_test, y_pred)

        acc.append(accu)
        folds.append(i)
        print(y_pred, "y values")

        print("accuracy: Fold",i+1, acc)

    plt.plot(folds, acc)
    plt.xlabel('Number of Folds')
    plt.ylabel('accuracy')
    plt.title('accuracy vs. Fold')
    plt.show()

def normalization(X_train, X_test):
    max_X_tr = X_train.max(axis=0)
    min_X_tr = X_train.min(axis=0)
    # min_X_te = np.min(X_test)
    # max_X_te = np.max(X_test)
    X_train_new=(X_train - min_X_tr)/(max_X_tr-min_X_tr)
    X_test_new=(X_test - min_X_tr)/(max_X_tr-min_X_tr)
    # print(X_train_new)
    # print(X_test_new)
    return X_train_new, X_test_new

def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def main():
    X, y = load_data()
    return k_fold_test(X, y)

if __name__ == '__main__':
    main()
