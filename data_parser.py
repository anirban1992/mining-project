
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import re


def parser_sgm_file(data):
    """
        Function to parse SGM file and separate the tags : Filter out the tags needed now here
        Takes in SGM file data as parameter

        Keyword Arguments:
            data: the sgm file data
    """

    parser = BeautifulSoup(data)

    content_body = parser.findAll('body')
    print "No of body tags:"
    print len(content_body)

    initial_sum = 0
    for lines in content_body:
        initial_sum += len(lines.text.split())
    print "Initial number of words in body: "
    print initial_sum

    # This strips the tag and retrieves the stemmed content of the tags body and title
    stemmed_body = []
    for lines_body in content_body:
        stemmed_body_row = tokenize_and_stem(lines_body.text)
        stemmed_body.append(stemmed_body_row)
    #print stemmed_body

    #No of words left after stemming
    stemmed_body_sum = 0
    for list_row_body in stemmed_body:
        stemmed_body_sum += len(list_row_body)
    print "Number of words from body after stemming and removing stop words:"
    print stemmed_body_sum

    content_title = parser.findAll('title')
    print "No of title tags: "
    print len(content_title)

    initial_sum = 0
    for lines in content_title:
        initial_sum += len(lines.text.split())
    print "Number of words in title tag:"
    print initial_sum

    stemmed_title = []
    for lines_title in content_title:
        stemmed_title_row = tokenize_and_stem(lines_title.text)
        stemmed_title.append(stemmed_title_row)
    #print stemmed_title

    #No of words left after stemming
    stemmed_title_sum = 0
    for list_row_title in stemmed_title:
        stemmed_title_sum += len(list_row_title)
    print "Number of words from title after stemming and removing stop words: "
    print stemmed_title_sum


def tokenize_and_stem(content):
    """
        Function to remove stop words and implement stemmer.
        Keyword Arguments:
            content: body of text to tokenize and stem
        Returns: list of stemmed root words
    """

    tokenizer = RegexpTokenizer(r'[A-Za-z\-]{2,}')
    tokens = tokenizer.tokenize(content.lower())
    good_words = [w for w in tokens if w.lower() not in stopwords.words('english')]
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(w) for w in good_words]
    return stemmed_words


def main():
    parser_sgm_file("dataset/data02.sgm")

if __name__ == "__main__":
    main()

