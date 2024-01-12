import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    dataset = pd.read_csv('./data/happiness.csv')
    print(dataset.describe)

    # Operating over the columns
    X = dataset[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
    y= dataset[['score']]

    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    modelLinear = LinearRegression().fit(X_train, y_train)
    y_predict_linear = modelLinear.predict(X_test)

    # Alpha is 1 by default. The bigger the alpha the higher the penalty
    modelLasso = Lasso(alpha=0.02).fit(X_train, y_train)
    y_predict_lasso = modelLasso.predict(X_test)

    modelRidge = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_ridge = modelRidge.predict(X_test)

    modelElastic = ElasticNet(random_state=0, alpha=0.005).fit(X_train, y_train)
    y_predict_elastic = modelElastic.predict(X_test)

    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print("Linear Loss: ", linear_loss)

    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print("Lasso Loss: ", lasso_loss)

    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print("Ridge Loss: ", ridge_loss)

    elastic_loss = mean_squared_error(y_test, y_predict_elastic)
    print("Elastic Loss: ", elastic_loss)

    print("="*32)
    print("Coef Lasso")
    print(modelLasso.coef_)

    print("="*32)
    print("Coef Ridge")
    print(modelRidge.coef_)

    print("="*32)
    print("Coef Ridge")
    print(modelElastic.coef_)

