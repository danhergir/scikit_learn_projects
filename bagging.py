from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import LinearSVC
from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    df = pd.read_csv('./data/heart.csv')
    print(df.target.describe())

    X = df.drop(['target'], axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

    ### Bagging is used for all of these estimators

    # estimators = {
    #     'KNeighbors': KNeighborsClassifier(),
    #     'LinearSCV': LinearSVC(max_iter=5000),
    #     'SVC': SVC(),
    #     'SGDC': SGDClassifier(),
    #     'DecisionTree': DecisionTreeClassifier(),
    #     'RandomTreeForest' : RandomForestClassifier(random_state=0)
    # }

    # for name, estimator in estimators.items():
    #     bag_class = BaggingClassifier(estimator=estimator, n_estimators=5).fit(X_train, y_train)
    #     bag_pred = bag_class.predict(X_test)

    #     print(f'Accuracy Bagging with {estimator}: ', accuracy_score(bag_pred, y_test))
    #     print('')

    ### Boosting is going to be used here
        
    # Playing with the estimators
    estimators = range(10, 200, 10)
    total_accuracy = []
    for i in estimators:
        boost = GradientBoostingClassifier(n_estimators=i).fit(X_train, y_train)
        boost_pred = boost.predict(X_test)

        total_accuracy.append(accuracy_score(y_test, boost_pred))
    
    plt.plot(estimators, total_accuracy)
    plt.xlabel('Estimators')
    plt.ylabel('Accuracy')
    plt.show()
    plt.savefig('Boost.png')

    print(np.array(total_accuracy).max())

