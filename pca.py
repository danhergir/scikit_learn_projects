import pandas as pd
import sklearn 
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    dt_heart = pd.read_csv('./data/heart.csv')
    
    print(dt_heart.head(5))

    # Column that we want to classify
    dt_features = dt_heart.drop(['target'], axis=1)
    # Our target means (presence of heart disease - TRUE/FALSE)
    dt_target = dt_heart['target']

    # Normalize data through StandardScaler
    dt_features = StandardScaler().fit_transform(dt_features)
    print(dt_features)

    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)

    print(X_train.shape)
    print(y_train.shape)

    # N_components is optional = min(n_muestras, n_features) = (n_rows, n_cols)
    pca = PCA(n_components=3)
    pca.fit(X_train)

    # Since IPCA does not train all the data, it uses batches to train them 
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)

    # Let's compare IPCA with PCA

    # From this plot, we can infere that the two first components are the ones which are contributing more information
    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.show()

    # Logistic Regression (Which is classification)

    # This is binary classification where we're trying to find out
    # if a pacient has or not a heart disease
    
    logistic = LogisticRegression(solver='lbfgs')

    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train, y_train)

    print("SCORE PCA: ", logistic.score(dt_test, y_test))

    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train, y_train)

    print("SCORE IPCA: ", logistic.score(dt_test, y_test))

    