from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from gensim.models import LdaModel
import logging
from gensim import corpora
import pandas
import numpy


stop_words = pandas.read_csv("stop_Words.txt", header=None)
stop_words = stop_words[0].tolist()

# scale an input array-like to a mininum and maximum number
# the input array must be of a floating point array
# if you have a non-floating point array, convert to floating using `astype('float')`
# this works with n-dimensional arrays
# it will mutate in place
# min and max can be integers
def scale_range (input, min, max):
    input += -(numpy.min(input))
    input /= numpy.max(input) / (max - min)
    input += min
    return input


def get_topics(raw_text, ngram=1, vocab_binary=True, nwords=30, ntopics=1):
	
    # Enable logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    print ('Number of documents: ' + str(len(raw_text)))
    print ('\nTokenizing documents..')

    tokens = []
    for doc in raw_text:
	
        try:
            doc = doc.encode("utf-8")
            token = word_tokenize(str(doc))

            clean_token = [i.lower() for i in token

                           if i.strip() not in stop_words
                           and i[:-2].strip() not in stop_words
                           and i[:-1].strip() not in stop_words
                           and i.strip().isalpha()
                           and len(i.strip()) > 1]

            ngram_tokens = []


            ngram_tokens.extend([x[0] for x in ngrams(clean_token, 1)])

            if ngram > 1:
                for i in range(2, ngram + 1):
                 ngram_tokens.extend([' '.join(x) for x in ngrams(clean_token, i)])

            if vocab_binary:
                tokens.append(set(ngram_tokens))
            else:
                tokens.append(ngram_tokens)

        except Exception as e:
            print (e)


    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(tokens)

    # dictionary.filter_extremes(no_above=1.0, keep_n=None)

    len(dictionary)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in tokens]

    lm = LdaModel(corpus=corpus, id2word=dictionary, passes=1,
                 num_topics=ntopics)

    # get topic 0 words (Topic: Default) for foam-tree viz
    words = (lm.show_topic(0, nwords))

    output = []

    weights = [x[1] for x in words]
    scaled_weights = scale_range(weights, 5,60)
    scaled_weights = numpy.nan_to_num(scaled_weights, 5)

    for i,word in enumerate(words):
        output.append({'label': word[0], 'weight': int(scaled_weights[i])})

    return  output
##################################################################################################################################
