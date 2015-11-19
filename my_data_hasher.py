from __future__ import division
import random
import time
import binascii
from bs4 import BeautifulSoup
import json
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt

topics_list=set()

cachedStopWords = set(stopwords.words("english"))

start = time.time()
article_list = {}  # documents with topic
article_info = {}

docs =0

for i in range(0,22):
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

        docs = docs +1

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

end = time.time()
numDocs = docs

with open('article_info.json', 'w') as fp:
    json.dump(article_info, fp)

def read_json_article():
    with open('article_info.json', 'r') as fp:
        article_info = json.load(fp)
    return article_info

def tokenize(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if (word not in cachedStopWords or topics_list) and len(word) > 3]
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

def sklearn_tdm_df(docs):
    '''
    Create a term-document matrix (TDM) in the form of a pandas DataFrame
    Uses sklearn's CountVectorizer function.

    Returns
    -------
    tdm_df: A pandas DataFrame with the term-document matrix. Columns are terms,
        rows are documents.
    '''
    # Initialize the vectorizer and get term counts in each document.
    vectorizer = CountVectorizer(tokenizer=tokenize,strip_accents='unicode',max_df=1.0,min_df=25)
    word_counts = vectorizer.fit_transform(docs.values())

    # .vocabulary_ is a Dict whose keys are the terms in the documents,
    # and whose entries are the columns in the matrix returned by fit_transform()
    vocab = vectorizer.vocabulary_

    # Make a dictionary of Series for each term; convert to DataFrame
    count_dict = {w: pd.Series(word_counts.getcol(vocab[w]).data) for w in vocab}
    tdm_df = pd.DataFrame(count_dict).fillna(0)

    return tdm_df


print "Shingling articles..."

curShingleID = 0
docsAsShingleSets = {};
docNames = []

t0 = time.time()

totalShingles = 0

for id,value in article_info.iteritems():

  #Getting the body of each document.Maintaining order so no stemming or tokenization done.
  words = unicodedata.normalize('NFKD', value['body']).encode('ascii','ignore')
  words = words.split(" ")
  docID = id
  docNames.append(docID)

  shinglesInDoc = set()

 # for index in range(0,len(words) - 4):
 # for index in range(0, len(words) - 2):
 # for index in range(0, len(words) - 1):
  for index in range(0,len(words)):
    # Using Shingle length = 1,2,3,5
    shingle = words[index]
    #shingle = words[index] + " " + words[index + 1]
    #shingle = words[index] + " " + words[index + 1] + " " + words[index + 2]
    #shingle = words[index] + " " + words[index + 1] + " " + words[index + 2] + words[index + 3] + " " + words[index + 4]

    # Hash the shingle to a 32-bit integer.
    crc = binascii.crc32(shingle) & 0xffffffff
    shinglesInDoc.add(crc)

  # Store the completed list of shingles for this document in the dictionary.
  docsAsShingleSets[docID] = shinglesInDoc

  totalShingles = totalShingles + len(words)
  #totalShingles = totalShingles + (len(words) - 1)
  #totalShingles = totalShingles + (len(words) - 2)
  #totalShingles = totalShingles + (len(words) - 4)

print '\nShingling ' + str(numDocs) + ' docs took %.2f sec.' % (time.time() - t0)

print '\nAverage shingles per doc: %.2f' % (totalShingles / numDocs)

numElems = int(numDocs * (numDocs - 1) / 2)

# 'JSim' will be for the actual Jaccard Similarity values.
# 'estJSim' will be for the estimated Jaccard Similarities found by comparing the MinHash signatures.
JSim = [0 for x in range(numElems)]
estJSim = [0 for x in range(numElems)]

def getTriangleIndex(i, j):
  if i == j:
    exit(1)
  if j < i:
    temp = i
    i = j
    j = temp

  k = int(i * (numDocs - (i + 1) / 2.0) + j - i) - 1

  return k

# Calculating the Jaccard similarities

print "\nCalculating Jaccard Similarities..."
t0 = time.time()

for i in range(0, numDocs):

    if (i % 1000) == 0:
        print "  (" + str(i) + " / " + str(numDocs) + ")"

    s1 = docsAsShingleSets[docNames[i]]

    for j in range(i + 1, numDocs):
        s2 = docsAsShingleSets[docNames[j]]

        if(len(s1.union(s2)) != 0):

            JSim[getTriangleIndex(i, j)] = (len(s1.intersection(s2)) / len(s1.union(s2)))

elapsed = (time.time() - t0)

print "\nCalculating all Jaccard Similarities took %.2fsec" % elapsed

# Record the maximum shingle ID that we assigned.
maxShingleID = 2**32-1


# Our random hash function will take the form of:
#   h(x) = (a*x + b) % c
# Where 'x' is the input value, 'a' and 'b' are random coefficients, and 'c' is
# a prime number just greater than maxShingleID.

def pickRandomCoeffs(k):
  randList = []

  while k > 0:
    randIndex = random.randint(0, maxShingleID)

    while randIndex in randList:
      randIndex = random.randint(0, maxShingleID)

    randList.append(randIndex)
    k = k - 1

  return randList

def compute_min_hash(numHashes):
    t0 = time.time()

    print '\nGenerating random hash functions...'

    # Largest prime number greater than 'maxShingleID'.
    # http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
    nextPrime = 4294967311
    coeffA = pickRandomCoeffs(numHashes)
    coeffB = pickRandomCoeffs(numHashes)

    print '\nGenerating MinHash signatures for all documents...'

    signatures = []


    for docID in docNames:

        shingleIDSet = docsAsShingleSets[docID]

        signature = []

        for i in range(0, numHashes):

            minHashCode = nextPrime + 1

            for shingleID in shingleIDSet:
                hashCode = (coeffA[i] * shingleID + coeffB[i]) % nextPrime
                if hashCode < minHashCode:
                    minHashCode = hashCode

            signature.append(minHashCode)

        signatures.append(signature)

    elapsed = (time.time() - t0)

    print "\nGenerating MinHash signatures took %.2fsec" % elapsed

    print '\nComparing all signatures...'

    t0 = time.time()

    for i in range(0, numDocs):
        signature1 = signatures[i]

        for j in range(i + 1, numDocs):

            signature2 = signatures[j]

            count = 0
            for k in range(0, numHashes):
                count = count + (signature1[k] == signature2[k])

            estJSim[getTriangleIndex(i, j)] = (count / numHashes)

    elapsed = (time.time() - t0)

    print "\nComparing MinHash signatures took %.2fsec" % elapsed


    threshold = 0.5
    # print "\nList of Document Pairs with J(d1,d2) more than", threshold
    # print "Values shown are the estimated Jaccard similarity and the actual"
    # print "Jaccard similarity.\n"
    # print "                   Est. J   Act. J"

    sum_squared_errors = 0

    for i in range(0, numDocs):
        for j in range(i + 1, numDocs):
            estJ = estJSim[getTriangleIndex(i, j)]
            J = JSim[getTriangleIndex(i,j)]
            err = abs((J-estJ)*(J-estJ))
            sum_squared_errors = sum_squared_errors + err

            # if estJ > threshold:
            #     #s1 = docsAsShingleSets[docNames[i]]
            #     #s2 = docsAsShingleSets[docNames[j]]
            #     #J = (len(s1.intersection(s2)) / len(s1.union(s2)))
            #     print "  %5s --> %5s   %.2f     %.2f" % (docNames[i], docNames[j], estJ, J)

    print "For k=",numHashes
    print "\nSum is :",sum_squared_errors
    print "\nSSE is:",sum_squared_errors/numDocs

    return float(sum_squared_errors/numDocs)

hash_range = [16,32,64,128,256]
sse = []
for k in hash_range:
    sse.append(compute_min_hash(k))

plt.plot(hash_range, sse, 'xb-')
plt.axis([5, 280, min(sse) - 1, max(sse) + 1])
plt.show()
