from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from time import time


def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]+', '', text)
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

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
                    article_info[article['newid']]['topic'].append(place.text)

                article_info[article['newid']]['label'] = article['lewissplit']

    # print 'No of valid articles = {}'.format(len(article_list))
    # print article_info

    vectorizer = TfidfVectorizer(min_df=0.01, stop_words=stopwords.words('english'), tokenizer=tokenize, strip_accents='unicode', smooth_idf=True)

    feature_vector = vectorizer.fit_transform(article_list)

    # print feature_vector

    print vectorizer.get_feature_names()
    print '\n'
    print len(vectorizer.get_feature_names())

    # bigram_vectorizer = TfidfVectorizer(min_df=0.01, stop_words=stopwords.words('english'), tokenizer=tokenize, ngram_range=(2,2), strip_accents='unicode', smooth_idf=True)

    print("Done in %0.3fs" % (time() - t0))

if __name__ == "__main__":
    main()

