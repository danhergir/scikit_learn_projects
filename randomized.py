import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    dataset = pd.read_csv('./data/happiness.csv')

    print(dataset)

    reg = RandomForestRegressor()

    X = dataset.drop(['country', 'rank', 'score'], axis=1)

    y = dataset[['score']].squeeze()

    params = {
        # Number of trees 
        'n_estimators': range(4, 16),
        'criterion': ['squared_error', 'absolute_error'],
        'max_depth': range(2, 11),
    }

    # It's going to try 10 different configs with the params given
    # 3 Folds
    rand_est = RandomizedSearchCV(reg, params, n_iter=10, cv = 3, scoring='neg_mean_absolute_error').fit(X, y)

    print(rand_est.best_estimator_)
    print(rand_est.best_params_)

    # Predicting for the first row of our dataset
    print(rand_est.predict(X.loc[[0]]))



