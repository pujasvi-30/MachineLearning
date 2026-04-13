import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def main():

    data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

    print("First 5 rows of dataset:")
    print(data.head())

    X = data.drop(["disease_score"],axis=1) # input features
    y = data["disease_score"]  # target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

    model = DecisionTreeRegressor(max_depth=3, random_state=10) # max_depth=3 limits tree height to avoid overfitting
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nPredicted values (first 5):")
    print(y_pred[:5])

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nMean Squared Error:", mse)
    print("R2 Score:", r2)

    # to plot tree structure
    plt.figure(figsize=(14, 6))
    plot_tree(model,feature_names=["BP", "age", "BMI", "blood_sugar"],filled=True)
    plt.title("Decision Tree Regressor")
    plt.show()

if __name__ == "__main__":
    main()