import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import (
    cross_val_score, KFold
)

if __name__ == '__main__':
    dataset = pd.read_csv('./data/happiness.csv')

    X = dataset.drop(['country', 'score'], axis=1)
    y = dataset['score']

    model = DecisionTreeRegressor()

    # cv = number of folds
    score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=3)
    # Absolute value of the mean
    print(np.abs(np.mean(score)))

    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    mse_values = []

    for train, test in kf.split(dataset):
        print(train)
        print(test)

        X_train = X.iloc[train]
        y_train = y.iloc[train]
        X_test = X.iloc[test]
        y_test = y.iloc[test]


        model = DecisionTreeRegressor().fit(X_train, y_train)
        predict = model.predict(X_test)
        mse_values.append(mean_squared_error(y_test, predict))

    print("Los tres MSE fueron: ", mse_values)
    print("El MSE promedio fue: ", np.mean(mse_values))

