import cPickle as pickle
import numpy as np
from time import time
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
from collections import Counter,Iterable
from sklearn import metrics
from sklearn.utils.extmath import density

predicted_dist = []
predicted_topic = []
X_train = []
X_test = []
y_train =[]
y_test_raw =[]
predictedClasses=[]
predictedClassesActual=[]
y_test = []

# The KNN classifier
def knn():
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(feature_vector, topic_labels,
                                                                         test_size=0.2, random_state=0)
    print X_train.shape
    X_train=np.array(X_train.toarray())
    X_test =np.array(X_test.toarray())

    print y_train
    print len(y_train)
    print y_train[282]

    t0 = time()
    neigh=NearestNeighbors(n_neighbors=12, algorithm='ball_tree').fit(X_train)
    train_time = time() - t0
    print("Offline Cost : The total time for knn: " +str(train_time), "seconds")

    t0 = time()
    for vector in X_test:
        distance, indices = neigh.kneighbors(vector)
        predicted_dist.append(distance)
        predicted_topic.append(getTopicsFromIndices(indices))
    test_time = time() - t0
    print("Online Cost : To predict from test set " +str(test_time), "seconds")

def getTopic(index):
    for inp, list in enumerate(y_train):
        print inp
        if (inp == index):
            for ind,value in enumerate(list):
                print "hi"
                return value

def flatten(lis):
     print lis
     for item in lis:
         if isinstance(item, Iterable) and not isinstance(item, basestring):
             for x in flatten(item):
                 yield x
         else:
             yield item


def getTopicsFromIndices(indexList):
    topicList=[]
    for indexes in indexList:
        for index in indexes:
            list_labels = list(flatten(y_train[index]))
            topicList.append(list_labels)
    return topicList

def getPredictedClasses():
    print(predicted_topic)
    for classes1, classes2 in zip(predicted_topic, y_test):
        classLabels=[]
        for class1, class2 in zip(classes1, classes2):
            for topic1, topic2 in zip(class1, class2):
                actualLength=(len(classes2))
                classLabels.append(topic1)
            predictedClasses.append(((Counter(classLabels)).most_common(actualLength)))
        for topics in predictedClasses:
            topicWithoutFrequency=set()
        for topic in topics:
            for subtopic in topic:
                if isinstance(subtopic, int):
                    pass
                else:
                    topicWithoutFrequency.add((subtopic))
        predictedClassesActual.append(list(topicWithoutFrequency))


def findAccuracy():
    success=0
    t0=time()
    for actualClass, predictedClass in zip(y_test, predictedClassesActual):
        if len(set(actualClass) & set(predictedClass))>0:
            success += 1
    print "total test"
    print len(y_test)
    print "success"
    print success
    accuracy = float((success*100)/len(y_test))
    print("The accuracy is %0.2f"%(accuracy))
    print("Online Cost : To predict from test set: " +str(time() - t0), "seconds")

def main():
    global feature_list
    feature_list = pickle.load(open('features_list', 'rb'))
    global feature_vector
    feature_vector = pickle.load(open('feature_vector', 'rb'))
    global topic_labels
    topic_labels = pickle.load(open('topic_labels','rb'))

    knn()
    getPredictedClasses()
    findAccuracy()

if __name__ == "__main__":
    main()
