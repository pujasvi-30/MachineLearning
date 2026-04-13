import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
import seaborn as sns

def load():
    data=load_iris()
    X=pd.DataFrame(data.data, columns=data.feature_names)
    y=pd.Series(data.target)
    y.fillna(y.mean(), inplace=True)
    print(data.data.shape)
    # print(X.columns)
    # print(y.shape)
    return X, y

def eda_data(X, y):
    print(y.value_counts()) #counts the freq of each class label in the target variable
    print("since all the classes have equal freq, the dataset is balanced, else would have been imbalanced")
    plt.hist(y)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    """Inference of X.hist()
    shape if bell-shaped: data is evenly distributed around mean
     right(+ve) or left(-ve)skew: data biased towards one feature, outliers present(outliers get high imp in adaboost)
      spread- wide(high feature variability and more info available
      clear separation- perfect predictor
      overlapping- weak predictor

      sepal length: moderately distributed, overlapping, moderate spread(weak predictor)
      sepal width: normal distributed, overlapping, narrow spread(weak predictor)
      petal length and width: normal distributed, clear separation, moderate spread(strong predictor): """
    X.hist(bins=20)
    plt.show()

    # corr = X.corr()
    # sns.heatmap(corr, annot=True)
    # plt.figure(figsize=(10, 5))
    # plt.title("Feature Correlation")
    # plt.show()


    df = pd.concat([X, y], axis=1)
    df['target'] = y
    sns.pairplot(pd.concat([X, y], axis=1), hue=y.name)
    plt.show()

    X.plot(kind='box', figsize=(10, 6))
    plt.title("Boxplot for Outlier Detection")
    plt.show()


def k_fold(X,y, k=10, shuffle=True):
    fold = KFold(n_splits=k, shuffle=shuffle, random_state=42)
    acc=[]
    folds=[]
    for f, (train_index, test_index) in enumerate (fold.split(X)):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        X_train_new, X_val, y_train_new, y_val= train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        depth=5
        best_depth=None
        best_val_score=-1
        for i in range(depth):
            model=train_model(X_train_new, y_train_new,i+1)
            model.fit(X_train_new, y_train_new)
            y_pred=model.predict(X_val)
            val_score=accuracy_score(y_val, y_pred)

            if val_score > best_val_score:
                best_val_score=val_score
                best_depth=i+1
        print(f"depth:{best_depth}, best_val_score:{best_val_score}")

        final_model = train_model(X_train, y_train, best_depth)
        final_model.fit(X_train, y_train)
        y_pred_final = final_model.predict(X_test)
        ac_score=accuracy_score(y_test, y_pred_final)
        acc.append(ac_score)
        folds.append(f+1)
        print(y_pred_final[:5])
        print("accuracy:", f+1, ac_score)
    return folds, acc

def train_model(X_train, y_train,best_depth):
    stump=DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    model= AdaBoostClassifier(estimator=stump, random_state=42, n_estimators=50, learning_rate=0.5) #by reducing n_estimators and learning_rate, cross validation overfitting can be controlled
    model.fit(X_train, y_train)
    return model

def plt_tree(folds, acc):
    plt.figure(figsize=(12, 8))
    plt.title('Adaboost Decision Tree Classifier')
    plt.xlabel('no.of folds')
    plt.ylabel('Accuracy')
    plt.plot(folds, acc)

def plot_main_tree(X, y):
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(X, y)
    plt.figure(figsize=(12, 8))
    plt.title('Decision Tree Classifier')
    plot_tree(tree, feature_names=X.columns,class_names=['setosa', 'versicolor', 'virginica'], filled=True)
    plt.show()

def main():
    X,y=load()
    eda_data(X,y)
    folds, acc= k_fold(X,y)
    plt_tree(folds, acc)
    plot_main_tree(X,y)

if __name__=='__main__':
    main()