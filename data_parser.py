from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from time import time
import numpy as np


def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]+', '', text)
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,indptr =array.indptr, shape=array.shape )

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

                topic_parser = article.topics
                place_parser = article.places

                for topic in topic_parser.findAll('d'):
                    article_info[article['newid']]['topic'].append(topic.text)

                for place in place_parser.findAll('d'):
                    article_info[article['newid']]['place'].append(place.text)

                article_info[article['newid']]['label'] = article['lewissplit']

        '''
        Extracting the dictionary of features into a .csv file
        Format :
            Article ID,Topic,Place, Label
            20057,[u'south-korea'],[],TEST
        '''
        with open('dictionary.csv', 'wb') as f:
            f.write('Article ID,Topic,Place,Label')
            f.write('\n')
            for key,value in article_info.iteritems():
                f.write(key)
                f.write(',')
                for inner_key,inner_value in value.items():
                    f.write(str(inner_value))
                    f.write(',')
                f.write('\n')

    # print 'No of valid articles = {}'.format(len(article_list))
    # print article_info

    with open('initial_word_count.txt','wb') as ini:
        sum =0
        for word in article_list:
            sum += len(word.split())
        ini.write('Total words in body tag of all the 21578 documents initially :'+str(sum))

    vectorizer = TfidfVectorizer(min_df=0.01, stop_words=stopwords.words('english'), tokenizer=tokenize, strip_accents='unicode', smooth_idf=True)

    feature_vector = vectorizer.fit_transform(article_list)

    with open('unigram_word_count.txt','wb') as ini:
            sum = len(vectorizer.get_feature_names())
            ini.write('Total words in body tag remaining after stemming , removing stop words and computing tf-idf counts :'+str(sum))

    # print feature_vector

    save_sparse_csr('feature_vector_unigram.csv',feature_vector)

    print vectorizer.get_feature_names()
    print '\n'
    print len(vectorizer.get_feature_names())

    bigram_vectorizer = TfidfVectorizer(min_df=0.01, stop_words=stopwords.words('english'), tokenizer=tokenize, ngram_range=(2,2), strip_accents='unicode', smooth_idf=True)

    bigram_feature_vector = bigram_vectorizer.fit_transform(article_list)

    indices = np.argsort(bigram_vectorizer.idf_)[::-1]
    features = bigram_vectorizer.get_feature_names()
    top_n = 20
    top_features = [features[i] for i in indices[:top_n]]
    print top_features
    with open('top_20_bigrams.txt','wb') as ini:
             ini.write(str(top_features))
    print("Done in %0.3fs" % (time() - t0))

if __name__ == "__main__":
    main()

