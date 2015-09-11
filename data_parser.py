from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def vectorize_document(document):
    """
    Function to compute the tf-idf
    :param document:
    :return:
    """
    #Vectorize the document to compute tf-idf
    document_tfidf_vectorizer = TfidfVectorizer(max_df=0.9,min_df=0.1)

    #fit the vectorizer to the document content
    tfidf_document_matrix = document_tfidf_vectorizer.fit_transform(document)
    print tfidf_document_matrix.shape
    idf = document_tfidf_vectorizer._tfidf.idf_
    print dict(zip(document_tfidf_vectorizer.get_feature_names(), idf))




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

    return initial_sum, stemmed_body, stemmed_body_sum


def parse_article(article):
    """
        Parses each article for its respective sub elements such as topics, titles, body etc

        Keyword Arguments:
            article_id: the new id of one reuters article
            article: contains exactly one reuters article
    """
    article_body = article.body
    article_topic = article.topics
    article_title = article.title
    article_date = article.date
    article_place = article.place

    #article_reuters = article.reuters['NEWID']
    #print article_reuters

    '''
    This is to just get the string values for non-null relevant tags
    '''


    if(article_body != None):
        initial_body_count, stemmed_body, final_body_count = retrieve_stemmed_content(article_body.text)
     #   print article_body.text
     #   print stemmed_body
        vectorize_document(stemmed_body)

    if(article_topic != None):
        initial_topic_count, stemmed_topic, final_topic_count = retrieve_stemmed_content(article_topic.text)

    if(article_title != None):
        initial_title_count, stemmed_title, final_title_count = retrieve_stemmed_content(article_title.text)

    if(article_place != None):
        initial_place_count, stemmed_place, final_place_count = retrieve_stemmed_content(article_place.text)

    print article_topic
    print article_date

def main():
    """
    Read sgm files and parse each article from the individual documents
    :return:
    """
    article_dict_collection = []
    for i in range(0,21):
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

            for article in parser.findAll('reuters'):
                article_id = article.get('newid')
                # print article_id
                article_dict = dict(article.attrs)
                if(article.body != None):
                    article_dict['Body'] = tokenize_and_stem(article.body.text)
                else:
                    article_dict['Body'] = ' '
                if(article.topics != None):
                    article_dict['Topics'] = tokenize_and_stem(article.topics.text)
                else:
                    article_dict['Topics'] = ' '
                if(article.title != None):
                    article_dict['Title'] = tokenize_and_stem(article.title.text)
                else:
                    article_dict['Title'] = ' '
                if(article.places != None):
                    article_dict['Places'] = tokenize_and_stem(article.places.text)
                else:
                    article_dict['Places'] = ' '
                if(article.date != None):
                    article_dict['Date'] = tokenize_and_stem(article.date.text)
                else:
                    article_dict['Date'] = ' '
                article.clear()

                article_dict_collection.append(article_dict)
    print len(article_dict_collection)

if __name__ == "__main__":
    main()

