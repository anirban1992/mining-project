from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.io.matlab.mio4 import arr_to_2d
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
import numpy as np
from scipy.sparse import dok_matrix
from scipy import *
from time import time
from sklearn.feature_extraction.text import CountVectorizer

def count_word_occurrence(article_body):
    """
    Function to get the number of occurrences of each word in an article
    :param article_body:
    :return: article_no_of_words : array of word count of the words occurring in each article taken as a document
    """
    try:
        word_count = CountVectorizer()
        article_no_of_words = word_count.fit_transform(article_body)
        # print type(article_no_of_words)
        # print article_no_of_words.toarray()
        print article_no_of_words
        return article_no_of_words
    except:
        return 0


def vectorize_document(document):
    """
    Function to compute the tf-idf
    :param document:
    :return:
    """
    #Vectorize the document to compute tf-idf
  #  document_tfidf_vectorizer = TfidfVectorizer(min_df=0.1)
    document_tfidf_vectorizer = TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)
    '''
    fit the vectorizer to the document content
    '''
    tfidf_document_matrix = document_tfidf_vectorizer.fit_transform(document)
   # print tfidf_document_matrix.idf
 #   print document_tfidf_vectorizer.idf_
   # print document_tfidf_vectorizer.__dict__
  #  print tfidf_document_matrix.shape
    #idf = document_tfidf_vectorizer._tfidf.idf_
   # print dict(zip(document_tfidf_vectorizer.get_feature_names(), idf))
  #  return dict(zip(document_tfidf_vectorizer.get_feature_names(), idf))



def tokenize_and_stem(content):
    """
        Function to remove stop words and implement stemmer.

        Keyword Arguments:
            content: body of text to tokenize and stem

        Returns: list of stemmed root words
    """

    # for cleaning the body of text
    content = re.sub('[^0-9a-zA-Z<>/\s=!-\"\"]+', '', content)  # cleans much better


    tokenizer = RegexpTokenizer(r'[A-Za-z\-]{2,}')
    tokens = tokenizer.tokenize(content.lower())
    good_words = [w for w in tokens if w.lower() not in stopwords.words('english')]
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(w) for w in good_words]
    return stemmed_words


def retrieve_stemmed_content(body_of_text):
    """
        Performs stemming and returns initial count, list of stemmed words and final count

        Keyword Arguments:
            body_of_text: text on which stemming is to be performed

        Returns:
            initial_sum: no of words initially
            stemmed_body: list of stemmed words
            stemmed_body_sum: after stemming the no of words
    """
    initial_sum = 0
    for lines in body_of_text:
        initial_sum += len(lines.split())

    # stemmed_body = []
    # for lines_body in body_of_text:
    #     stemmed_body_row = tokenize_and_stem(lines_body)
    #     stemmed_body.append(stemmed_body_row)

    stemmed_body = tokenize_and_stem(body_of_text)

    stemmed_body_sum = 0
    for list_row_body in stemmed_body:
        stemmed_body_sum += len(list_row_body)

    return stemmed_body

def main():
    """
    Read sgm files and parse each article from the individual documents
    :return:
    """
    document_word_count = []
    t0 = time()
    for i in range(0,22):
        filename = 'data{}'.format(str(i).zfill(2))
        with open('dataset/'+filename+'.sgm' , 'r') as f:
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


            '''
            For now just iterating through all the documents , getting body from each article
            computing tf-idf for each body and removing stop words
            Resulting in a dictionary of the parsed words from the body of each article with the corresponding tf-idf.

            '''

            for article in parser.findAll('reuters'):
                article_id = article.get('newid')
                print article_id
                article_dict = dict(article.attrs)
                if(article.body != None):
                    stemmed_body = tokenize_and_stem(article.body.text)
                    document_word_count.append(count_word_occurrence(stemmed_body))
                article.clear()
            break
    print("Done in %0.3fs" % (time() - t0))

if __name__ == "__main__":
    main()

