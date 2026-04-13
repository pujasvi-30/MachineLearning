import itertools
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from ISLP import load_data

def load():
    data = load_data('Weekly')
    print(data.shape)
    print(data.columns)
    X = data.drop('Direction', axis=1)
    y = data['Direction']
    return X,y

def split_data(X,y):
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42) #validation set split
    return X_train, X_test, y_train, y_test, X_val, y_val, X_tr, y_tr

def main():
    X, y = load()
    X_train, X_test, y_train, y_test, X_val, y_val, X_tr, y_tr = split_data(X,y)
    n_estimators_range = [50, 100, 150]
    learning_rate_range = [0.01, 0.05, 0.1]
    max_depth_range = [3, 4, 5]
    reg_alpha_range = [0, 0.01, 0.1]

    best_acc = float('inf')  # to initialize best accuracy score
    best_params = None

    for n_estimators, lr, max_depth, reg_alpha in itertools.product(
            n_estimators_range, learning_rate_range, max_depth_range, reg_alpha_range):
        #itertools.product is to get cartesian product/ all combinations of parameters and ranges
        model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=lr,
            max_depth=max_depth,
            reg_alpha=reg_alpha,
            eval_metric='logloss',
            objective='binary:logistic',
            random_state=42
        )
        model.fit(X_tr, y_tr)
        y_val_pred = model.predict(X_val)
        acc= accuracy_score(y_val, y_val_pred)

        if acc < best_acc:
            best_acc = acc
            best_params = {
                'n_estimators': n_estimators,
                'learning_rate': lr,
                'max_depth': max_depth,
                'reg_alpha': reg_alpha
            }
    print("Best Hyperparameters:", best_params)
    print("Best Validation accuracy:", best_acc)

    # train final model on the entire training data with best params
    final_model = XGBRegressor(
        **best_params,
        objective='reg:squarederror',
        random_state=42
    ) #**best_params will unpack dictionary into keyword arg
    final_model.fit(X_train, y_train)
    y_test_pred = final_model.predict(X_test)
    print("Test R2:", r2_score(y_test, y_test_pred))
    print("Test MSE:", mean_squared_error(y_test, y_test_pred))


if __name__ == '__main__':
    main()