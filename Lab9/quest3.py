

import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def main():

    df = pd.read_csv('sonar data.csv')
    X = df.iloc[:, 0:59]
    Y = df.iloc[:, 60]
    y = Y.map({'R': 0, 'M': 1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = DecisionTreeClassifier(criterion='entropy',max_depth=5, random_state=42) # criterion='entropy' uses Information Gain
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    print("Accuracy:",accuracy)
    print("Classification report:\n",classification_report(y_test,y_pred))

    plt.figure(figsize=(12,6))
    plot_tree(model,filled=True)
    plt.show()

if __name__ == '__main__':
    main()