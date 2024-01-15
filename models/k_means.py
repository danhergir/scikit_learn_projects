import pandas as pd

from sklearn.cluster import MiniBatchKMeans

if __name__ == '__main__':
    dataset = pd.read_csv('./in/candy.csv')
    print(dataset.head(10))

    # Since this is unsupervised learning, test and train data split is not required.
    X = dataset.drop('competitorname', axis=1)

    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)
    print("Total Centroids: ", len(kmeans.cluster_centers_))

    print('='*32)
    print(kmeans.predict(X))

    dataset['group'] = kmeans.predict(X)

    print(dataset)
