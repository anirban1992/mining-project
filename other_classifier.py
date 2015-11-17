from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import *
from sklearn.tree import *
import cPickle as pickle
import arrow
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from nltk.classify import DecisionTreeClassifier
import numpy as np

# MAJOR IMPORTANT PART
# feature_vector_train , feature_vector_test -> tuples of (feature type, topic)

# Feature vector for training
# features_train, topics_train -> extract from feature_vector_train
# where features would be list of feature types and topics would be list of topics -> for each

# Feature vector for test
# features, topics -> extract from feature_vector_test
# where features would be list of feature types and topics would be list of topics -> for each


def create_training_test_data(feature_dict):
    # structure is
    # { topic : { [{term : 1 , term2 : 1} , {term1: 2, term3 : 4}] }, topic2 : {[{}, {}]}
    SPLIT_RATIO = 0.8
    train_data = []
    test_data = []
    for topic, topic_vector in feature_dict.iteritems():
        train_limit = len(topic_vector) * SPLIT_RATIO
        for i, feature_vector in enumerate(topic_vector):
            if i <= train_limit:
                train_data.append([feature_vector, topic])
            else:
                test_data.append([feature_vector, topic])
    return train_data, test_data


def naive_bayes_classifer(feature_vector_train, feature_vector_test):
    vectorizer = DictVectorizer(dtype=float, sparse=True)
    classifier = MultinomialNB()

    features_train, topics_train = zip(*feature_vector_train)
    features_test, topics_test = zip(*feature_vector_test)

    # important to note everything should be array like
    # So we have to use np.array mostly

    # training
    features_1 = vectorizer.fit_transform(np.array(features_train))
    classifier.fit(features_1, topics_train)

    # testing

    features_2 = vectorizer.transform(np.array(features_test))

    predicted_topics = classifier.predict(features_2)

    # One awesome command in sklearn
    print classification_report(topics_test, predicted_topics, target_names=set(topics_test))


def decision_tree_classifier(feature_vector_train, feature_vector_test):
    features_train, topics_train = zip(*feature_vector_train)
    features_test, topics_test = zip(*feature_vector_test)

    # training

    classifier2 = DecisionTreeClassifier.train(features_train, depth_cutoff=250, entropy_cutoff=0.1)

    # Kept an entropy cutoff in order to improve the training time (this might lead to loss in accuracy though)
    # Same goes for depth cutoff (for refining the tree). Kept it as 250.

    # testing

    predicted_topics = classifier2.classify_many(features_test)

    print classification_report(topics_test, predicted_topics, target_names=set(topics_test))

def main():
    # get the feature_vector_train and feature_vector_test.. Your choice of splitting
    feature_dict = pickle.load(open('features_list', 'rb'))

    feature_vector_train, feature_vector_test = create_training_test_data(feature_dict)

    start_time1 = arrow.utcnow().timestamp
    naive_bayes_classifer(feature_vector_train, feature_vector_test)
    end_time1 = arrow.utcnow().timestamp

    start_time2 = arrow.utcnow().timestamp
    decision_tree_classifier(feature_vector_train, feature_vector_test)
    end_time2 = arrow.utcnow().timestamp

    print 'Time taken for naive bayes is {} seconds'.format(end_time1 - start_time1)

    print 'Time taken for decision tree is {} seconds'.format(end_time2 - start_time2)

if __name__ == '__main__':
    main()