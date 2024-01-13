import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == '__main__':
    dataset = pd.read_csv('./data/candy.csv')
    print(dataset.head(5))

    X = dataset.drop('competitorname', axis=1)

    meanshift = MeanShift().fit(X)
    
    # Number of labels based on max number
    print('Number of labels: ', max(meanshift.labels_))

    # Centroids
    print('Centroids: ', meanshift.cluster_centers_)

    dataset['meanshift'] = meanshift.labels_

    print('='*32)
    print(dataset)


