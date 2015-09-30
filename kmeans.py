import cPickle as pickle
import numpy as np
from time import time
from sklearn.cluster import KMeans,MiniBatchKMeans

cls = open('clusters.csv','wb')

def kMeans(num_clusters):
    """
    Runs K-Means clustering on the feature_vector and list
    :param num_clusters:
    :return:
    """
    batch_size = 50
    km = KMeans(init='k-means++',n_clusters=num_clusters,verbose=True)

    t0 = time()
    km.fit(feature_vector,feature_list)
    offline_cost = time() - t0

    clusters = km.labels_.tolist()
   # print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    for i in np.unique(clusters):
        cls.write('Cluster ')
        cls.write(str(i))
        cls.write(' : ')
        for ind in order_centroids[i, :10]:
            cls.write(feature_list[ind])
            cls.write(' ')
    cls.write(',')
    cls.write(str(offline_cost))
    cls.write(',')

    #To take batch of data : to compute online cost for a tuple of the features. # Could be improved here
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=num_clusters, batch_size=batch_size,verbose=True)
    t0 = time()
    mbk.fit(feature_vector)
    online_cost = time() - t0

    cls.write(str(online_cost))

def main():
    """
    Loads the feature vector and feature list.
    Runs K-Means clustering on the feature_vector and list with different cluster numbers.
    :return:
    """
    global feature_list
    feature_list = pickle.load(open('features_list', 'rb'))
    print(feature_list)
    global feature_vector
    feature_vector = pickle.load(open('feature_vector', 'rb'))
    cls.write('Number of clusters,Clusters formed , Offline Cost, Online Cost')
    cls.write('\n')
    for i in (3,6,10,12,15):
        cls.write(str(i))
        cls.write(',')
        kMeans(i)
        cls.write('\n')


if __name__ == "__main__":
    main()

