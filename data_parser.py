from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re


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

    stemmed_body = []
    for lines_body in body_of_text:
        stemmed_body_row = tokenize_and_stem(lines_body)
        stemmed_body.append(stemmed_body_row)

    stemmed_body_sum = 0
    for list_row_body in stemmed_body:
        stemmed_body_sum += len(list_row_body)

    return initial_sum, stemmed_body, stemmed_body_sum


def parse_article(article):
    """
        Parses each article for its respective sub elements such as topics, titles, body etc

        Keyword Arguments:
            article: contains exactly one reuters article
    """
    article_body = article.body.string
    article_topic = article.topics.string
    article_title = article.title.string
    article_date = article.date.string

    initial_body_count, stemmed_body, final_body_count = retrieve_stemmed_content(article_body)

    initial_title_count, stemmed_title, final_title_count = retrieve_stemmed_content(article_title)

    print article_topic
    print article_date


def parser_sgm_file(data):
    """
        Function to parse SGM file and separate the tags : Filter out the tags needed now here
        Takes in SGM file data as parameter

        Keyword Arguments:
            data: the sgm file data
    """

    parser = BeautifulSoup(data, 'html.parser')

    print len(parser.findAll('reuters'))  # this contains the no of articles in one sgm file, should be 1000

    for article in parser.findAll('reuters'):
        parse_article(article)
        break


def main():
    with open('dataset/data02.sgm', 'r') as f:
        data = f.read()

    parser_sgm_file(data)

if __name__ == "__main__":
    main()

