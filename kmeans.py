import cPickle as pickle
import numpy as np
from time import time
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn import cross_validation
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab as pl

cls = open('clusters.csv','wb')
inertia = []
fig = plt.figure()
plots = []
legends = []

def kMeans(num_clusters):
    """
    Runs K-Means clustering on the feature_vector and list
    :param num_clusters:
    :return:
    """
    batch_size = 50
    km = KMeans(init='random',n_clusters=num_clusters,random_state=1)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(feature_vector, topic_labels,
                                                                         test_size=0.2, random_state=0)
    t0 = time()
    km.fit(X_train)
    offline_cost = time() - t0

    clusters = km.labels_.tolist()
   # print("Top terms per cluster:")
    order_centroids_kmeans = km.cluster_centers_.argsort()[:, ::-1]
    for i in np.unique(clusters):
        cls.write('Cluster ')
        cls.write(str(i))
        cls.write(' : ')
        for ind in order_centroids_kmeans[i, :10]:
            cls.write(feature_list[ind])
            cls.write(' ')
    cls.write(',')
    cls.write(str(offline_cost))
    cls.write(',')

    #To take batch of data : to compute online cost for a tuple of the features. # Could be improved here
    mbk = MiniBatchKMeans(init='random', n_clusters=num_clusters,random_state=1,n_init=1)
    t0 = time()
    miniK = mbk.fit(X_train)

    online_cost = time() - t0
    cls.write(str(online_cost))

    order_centroids_mbk = mbk.cluster_centers_.argsort()[:, ::-1]

    # Plot result

    fig = pl.figure()
    colors = ['#4EACC5', '#FF9C34', '#4E9A06']

    # The cluster centers as per closest one in MiniBatchKMeans and the KMeans algorithm..

    distance = euclidean_distances(order_centroids_kmeans,order_centroids_mbk,squared=True)
    order = distance.argmin(axis=1)

    # KMeans
    ax = fig.add_subplot(1, 3, 1)
    for k in range(num_clusters):
        col = cm.spectral(float(k) / num_clusters, 1)
        my_members = km.labels_ == k
        cluster_center = km.cluster_centers_[k]
        ax.plot(X_train[my_members, 0], X_train[my_members, 1], 'w',markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=6)
    ax.set_title('KMeans')
    pl.text(-3.5, 2.7,  'train time: %.2fs' % offline_cost)

    # MiniBatchKMeans
    ax = fig.add_subplot(1, 3, 2)
    for k in range(num_clusters):
        col = cm.spectral(float(k) / num_clusters, 1)
        my_members = mbk.labels_ == order[k]
        cluster_center = mbk.cluster_centers_[order[k]]
        ax.plot(X_train[my_members, 0], X_train[my_members, 1], 'w',markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=6)
    ax.set_title('MiniBatchKMeans')
    pl.text(-3.5, 2.7,  'train time: %.2fs' % online_cost)

    different = (mbk.labels_ == 4)
    ax = fig.add_subplot(1, 3, 3)

    for l in range(num_clusters):
        different += ((km.labels_ == k) != (mbk.labels_ == order[k]))

    identic = np.logical_not(different)
    ax.plot(X_train[identic, 0], X_train[identic, 1], 'w',markerfacecolor='#bbbbbb', marker='.')
    ax.plot(X_train[different, 0], X_train[different, 1], 'w',markerfacecolor='m', marker='.')
    ax.set_title('Difference')

    pl.show()

def main():
    """
    Loads the feature vector and feature list.
    Runs K-Means clustering on the feature_vector and list with different cluster numbers.
    :return:
    """
    global feature_list
    feature_list = pickle.load(open('features_list', 'rb'))
    global feature_vector
    feature_vector = pickle.load(open('feature_vector', 'rb'))
    global topic_labels
    topic_labels = pickle.load(open('topic_labels','rb'))

    feature_vector = feature_vector.toarray()

    cls.write('Number of clusters,Clusters formed , Offline Cost, Online Cost')
    cls.write('\n')
    for i in (3,6,10,12,15):
        cls.write(str(i))
        cls.write(',')
        kMeans(i)
        cls.write('\n')


if __name__ == "__main__":
    main()

