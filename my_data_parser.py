from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import OrderedDict
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer , TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.cluster import v_measure_score, silhouette_score
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from nltk.stem.porter import PorterStemmer
from collections import Counter
import math
from time import time
import json


option = 1
cachedStopWords = set(stopwords.words("english"))
topics_list = set()
article_info = {}


def tokenize(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if (word not in cachedStopWords or topics_list) and len(word) > 3]
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems


def write_json_article():
    global topics_list

    start = time()
    article_list = {}  # documents with topic
    article_info = {}

    for i in range(0, 22):
        filename = 'data{}'.format(str(i).zfill(2))
        with open('dataset/{}.sgm'.format(filename), 'r') as f:
            data = f.read()

        parser = BeautifulSoup(data, 'html.parser')
        for article in parser.findAll('reuters'):
            try:
                text_body = article.body.text
            except AttributeError:
                continue

            article_info[article['newid']] = {}
            article_info[article['newid']]['body'] = text_body
            article_info[article['newid']]['topic'] = []
            article_info[article['newid']]['place'] = []

            place_parser = article.places
            topic_parser = article.topics
            topic_list = []

            for topic in topic_parser.findAll('d'):
                topics_list.update(topic.text)
                topic_list.append(topic.text)

            for place in place_parser.findAll('d'):
                article_info[article['newid']]['place'].append(place.text)

            article_info[article['newid']]['label'] = article['lewissplit']
            if len(topic_list) != 0:
                article_info[article['newid']]['topic'].append(topic_list)

    end = time()
    with open('article_info.json', 'w') as fp:
        json.dump(article_info, fp)


def read_json_article():
    with open('article_info.json', 'r') as fp:
        article_info = json.load(fp)
    return article_info


def entropy(labels):
    p, lns = Counter(labels), float(len(labels))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())


def cluster_entropy(labels):
    cluster_topics_values = {}
    entropy_cluster = {}
    for i, label in enumerate(labels):
        id = article_dict_1.keys()[i]
        if label not in cluster_topics_values:
            cluster_topics_values[label] = []
        for topic in article_info[id]['topic']:
            cluster_topics_values[label].extend(topic)

    no_entropy_zero_clusters = 0
    no_entropy_one_greater_clusters = 0

    for label, value in cluster_topics_values.iteritems():
        entropy_cluster[label] = entropy(value)
        if entropy_cluster[label] > 1:
            no_entropy_one_greater_clusters += 1
        elif entropy_cluster[label] == 0:
            no_entropy_zero_clusters += 1

    print 'No of clusters with entropy greater than 1 = {} and no of clusters with entropy equal to 0 = {}'.format(
        no_entropy_one_greater_clusters, no_entropy_zero_clusters)


def my_dbscan(feature_vector, metric_name, eps=None, minpts=None):
    start = time()
    if eps is None and minpts is None:
        db = DBSCAN(metric=metric_name).fit(feature_vector)
    elif minpts is None:
        db = DBSCAN(eps=eps, metric=metric_name).fit(feature_vector)
    elif eps is None:
        db = DBSCAN(min_samples=minpts, metric=metric_name).fit(feature_vector)
    else:
        db = DBSCAN(eps=eps, min_samples=minpts, metric=metric_name).fit(feature_vector)
    end = time()
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # ignoring noise if present
    print 'The no of non noisy clusters is {} with metric = {}'.format(n_clusters, metric_name)
    print "Time taken to finish {} seconds".format(end - start)
    if option == 1:
        cluster_entropy(labels)
    else:
        print 'The silhouette score is {}'.format(silhouette_score(feature_vector, labels, metric=metric_name))


def my_kmeans(feature_vector, no_of_centers=8):
    start = time()
    km = KMeans(n_clusters=no_of_centers).fit(feature_vector)
    end = time()
    labels = km.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print 'The no of non noisy clusters is {} with no of centers = {}'.format(n_clusters, no_of_centers)
    print "Time taken to finish {} seconds".format(end - start)
    if option == 1:
        cluster_entropy(labels)
    else:
        print 'The silhouette score is {}'.format(silhouette_score(feature_vector, labels, metric='euclidean'))


def my_agg_clustering(feature_vector, no_of_centers, metric_name):
    start = time()
    ag_c = AgglomerativeClustering(n_clusters=no_of_centers, affinity=metric_name).fit(feature_vector)
    end = time()
    labels = ag_c.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print 'The no of non noisy clusters is {} with no of centers = {} with metric = {}'.format(
        n_clusters, no_of_centers, metric_name)
    print "Time taken to finish {} seconds".format(end - start)
    if option == 1:
        cluster_entropy(labels)
    else:
        print 'The silhouette score is {}'.format(silhouette_score(feature_vector, labels, metric=metric_name))


def dbscan_tester(feature_vector):
    my_dbscan(feature_vector, 'manhattan', 10)
    my_dbscan(feature_vector, 'euclidean')


def kmeans_tester(feature_vector):
    my_kmeans(feature_vector)
    my_kmeans(feature_vector, 20)
    my_kmeans(feature_vector, len(topics_list))

def agg_clustering_tester(feature_vector):
    my_agg_clustering(feature_vector, len(topics_list), 'euclidean')


def cluster_vectorizer(article_dict):
    vectorizer = TfidfVectorizer(min_df=25, max_df=1.0, tokenizer=tokenize, strip_accents='unicode', smooth_idf=True)

    feature_vector = vectorizer.fit_transform(article_dict.values())
    feature_list = vectorizer.get_feature_names()

    print 'The no of features is {}'.format(len(feature_list))

    feature_vector = feature_vector.todense()
    print '\nFor Kmeans algorithm'
    kmeans_tester(feature_vector)
    print '\nFor DBSCAN algorithm'
    dbscan_tester(feature_vector)
    print '\nFor hierarchical clustering (based on ward)'
    agg_clustering_tester(feature_vector)


def main():
    """
    Read sgm files and parse each article from the individual documents
    :return:
    """
    global article_info
    global article_dict_1
    global article_dict_2

    write_json_article()

    # print topics_list
    print len(topics_list)
    article_info = read_json_article()

    article_dict_1 = OrderedDict()
    article_dict_2 = OrderedDict()

    for id, value in article_info.iteritems():
        if value['topic']:
            article_dict_1[id] = value['body']
        else:
            article_dict_2[id] = value['body']

    print '\nFOR ARTICLES WITH TOPICS\n'
    cluster_vectorizer(article_dict_1)

    global option
    option = 2

    print '\nFOR ARTICLES WITHOUT TOPICS\n'
    cluster_vectorizer(article_dict_2)


if __name__ == "__main__":
    main()