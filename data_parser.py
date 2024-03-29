from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer , TfidfTransformer
from nltk.stem.porter import PorterStemmer
from time import time
import numpy as np
import cPickle as pickle

from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans

cachedStopWords = set(stopwords.words("english"))

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]+', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if (word not in cachedStopWords or topics_list) and len(word) > 3]
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

def getTopics(article_info,list):
    """
    Returns topic list.
    :param article_info:
    :param list:
    :return:
    """
    for key,value in article_info.iteritems():
        if isinstance(value,dict):
            getTopics(value,list)# If value itself is dictionary
        elif (key == 'topic'):
            list.append(value)
            # print list
    return list

def main():
    """
    Read sgm files and parse each article from the individual documents
    :return:
    """
    t0 = time()
    article_list = []
    article_info = {}

    for i in range(0, 22):
        filename = 'data{}'.format(str(i).zfill(2))
        with open('dataset/{}.sgm'.format(filename), 'r') as f:
            data = f.read()
        parser = BeautifulSoup(data, 'html.parser')
        '''
        Looping over each article distinguished by reuters tag , creating a dictionary out of each article of the format :
        {
            'Body': [u'jaguar', u'jaguar', u'plc', u'jagrl', u'sell', u'new', u'xj', u'model', u'us', u'japanes' ],
            'Places': [u'uk'],
            'Title': [u'jaguar', u'see', u'strong', u'growth', u'new', u'model', u'sale'],
            'Topics': [u'earn'],
            u'topics': u'YES',
            u'lewissplit': u'TRAIN',
            u'newid': u'2001',
            u'oldid': u'18419',
            'Date': [u'mar'],
            u'cgisplit': u'TRAINING-SET'
        }

        The content of each dictionary tag is after removing stop words and stemming the contents
        '''

        for article in parser.findAll('reuters'):
            try:
                article_list.append(article.body.text)

            except AttributeError:
                continue

            article_info[article['newid']] = {}
            article_info[article['newid']]['topic'] = []
            article_info[article['newid']]['place'] = []

            place_parser = article.places
            topic_parser = article.topics
            topic_list = []
            for topic in topic_parser.findAll('d'):
                topic_list.append(topic.text)

            for place in place_parser.findAll('d'):
                article_info[article['newid']]['place'].append(place.text)

            article_info[article['newid']]['label'] = article['lewissplit']
            if(len(topic_list)==0):
                article_list.pop()
                article_info.popitem()
            else:
                article_info[article['newid']]['topic'].append(topic_list)


        '''
        Extracting the dictionary of features into a .csv file
        Format :
            Article ID,Topic,Place, Label
            20057,[u'south-korea'],[],TEST
        '''

    # with open('dictionary.csv', 'wb') as f:
    #     f.write('Article ID,Topic,Place,Label')
    #     f.write('\n')
    #     for key, value in article_info.iteritems():
    #         f.write(key)
    #         f.write(',')
    #         for inner_key,inner_value in value.items():
    #             f.write(str(inner_value))
    #             f.write(',')
    #         f.write('\n')

    print 'No of valid articles = {}'.format(len(article_list))
    #To create a global list of topics used while tokenization(used in tokenize function) to make sure feature words do not belong to topic list
    global topics_list
    topics_list = list()
    topics = getTopics(article_info,[])

    for topic_article in topics:
        if topic_article:
            for topic in topic_article:
                if topic:
                    for top in topic:
                        topics_list.append(top)


    with open('topic_labels', 'wb') as outfile:
        pickle.dump(topics, outfile, pickle.HIGHEST_PROTOCOL)

        # with open('initial_word_count.txt', 'wb') as ini:
        #     sum =0
        #     for word in article_list:
        #         sum += len(word.split())
        #  ini.write('Total words in body tag of all the 21578 documents initially :'+str(sum))


    vectorizer = TfidfVectorizer(min_df= 0.001,max_df=0.9, tokenizer=tokenize, strip_accents='unicode', smooth_idf=True)

    feature_vector = vectorizer.fit_transform(article_list)

    feature_list = vectorizer.get_feature_names()

    with open('feature_vector', 'wb') as outfile:
        pickle.dump(feature_vector, outfile, pickle.HIGHEST_PROTOCOL)

    with open('features_list', 'wb') as f:
        pickle.dump(feature_list, f, pickle.HIGHEST_PROTOCOL)

# with open('feature_list.csv','wb') as feature:
#     for value in feature_list:
#         feature.write(str(value)+'\n')

    counter_vectorizer = CountVectorizer(vocabulary=vectorizer.vocabulary_, strip_accents='unicode')

    # for the word frequency counts
    data_matrix = counter_vectorizer.fit_transform(article_list)  # data matrix
    transaction_matrix = vectorizer.inverse_transform(feature_vector)  # transaction matrix
    # terms = counter_vectorizer.get_feature_names()
    # freqs = data_matrix
    # result = dict(zip(terms, freqs))
    # print result
    # print(len(result))



## Un-comment from here to generate data_matrix and transaction_matrix

# with open('data_matrix.dat', 'wb') as outfile:
#     pickle.dump(data_matrix, outfile, pickle.HIGHEST_PROTOCOL)
#
# with open('transaction_matrix.dat', 'wb') as outfile:
#     pickle.dump(transaction_matrix, outfile, pickle.HIGHEST_PROTOCOL)

# with open('unigram_word_count.txt','wb') as ini:
#         sum = len(vectorizer.get_feature_names())
#         ini.write('Total words in body tag remaining after stemming , removing stop words and computing tf-idf counts :'+str(sum))

    bigram_vectorizer = TfidfVectorizer(min_df=0.001, tokenizer=tokenize, ngram_range=(2,2), strip_accents='unicode', max_df=0.9, smooth_idf=True)

    bigram_feature_vector = bigram_vectorizer.fit_transform(article_list)

    indices = np.argsort(bigram_vectorizer.idf_)[::-1]
    features = bigram_vectorizer.get_feature_names()
    top_n = 20
    top_features = [features[i] for i in indices[:top_n]]
    print top_features
    # with open('top_20_bigrams.txt','wb') as ini:
    #          ini.write(str(top_features))
    print("Done in %0.3fs" % (time() - t0))

if __name__ == "__main__":
    main()

