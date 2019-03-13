# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:50:34 2017

@author: yyang
"""


import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *

from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.ldamodel import LdaModel

import json
import pyLDAvis.gensim

from sklearn.feature_extraction.text import CountVectorizer 

from gensim.models import word2vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


###############################################################################
'''data preprocess'''
###############################################################################
yelp_comp = pd.read_csv("completeYelpData.csv", encoding = "ISO-8859-1")

yelp_sub = yelp_comp[["text", 'rating', 'id']]

yelp_sub = yelp_sub.dropna(axis=0, how='any')
label = [1 if w >=4 else 0 for w in yelp_sub['rating']]
text = yelp_sub['text']
 
def preprocess_text(corpus):
    """Takes a corpus in list format and applies basic preprocessing steps of word tokenization,
     removing of english stop words, lower case and lemmatization."""
    processed_corpus = []
    english_words = set(nltk.corpus.words.words())
    english_stopwords = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer(r'[\w|!]+')
    for row in corpus:
        word_tokens = tokenizer.tokenize(row)
        word_tokens_lower = [t.lower() for t in word_tokens]
        word_tokens_lower_english = [t for t in word_tokens_lower if t in english_words or not t.isalpha()]
        word_tokens_no_stops = [t for t in word_tokens_lower_english if not t in english_stopwords]
        word_tokens_no_stops_lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in word_tokens_no_stops]
        word_tokens_no_stops_lemmatized_stem = [stemmer.stem(t) for t in word_tokens_no_stops_lemmatized]
        processed_corpus.append(word_tokens_no_stops_lemmatized_stem)
    return processed_corpus

text.head()
corpus_description = text.astype(str)
processed_corpus_description = preprocess_text(corpus_description)



###############################################################################
''' word cloud'''
###############################################################################
$ pip install git+git://github.com/amueller/word_cloud.git

wordcloud = WordCloud().generate(text)
img=plt.imshow(wordcloud)
plt.axis("off")
plt.show()


###############################################################################
'''topic modeling'''
###############################################################################


def nlp_model_pipeline(processed_corpus):
    """Takes processed corpus and produce dictionary, doc_term_matrix and LDA model"""
    # Creates the term dictionary (every unique term in corpus is assigned an index)
    dictionary = Dictionary(processed_corpus)
    # Convert corpus into Document Term Matrix using dictionary prepared above
    doc_term_matrix = [dictionary.doc2bow(listing) for listing in processed_corpus]    
    return dictionary, doc_term_matrix


def LDA_topic_modelling(doc_term_matrix, dictionary, num_topics=3, passes=2):
    # Create an object for LDA model and train it on Document-Term-Matrix
    LDA = LdaModel
    ldamodel = LDA(doc_term_matrix, num_topics=num_topics, id2word = dictionary, passes=passes)
    return ldamodel

def add_topics_to_df(ldamodel, doc_term_matrix, df, new_col, num_topics):
    # Convert into Per-document topic probability matrix:
    docTopicProbMat = ldamodel[doc_term_matrix]
    docTopicProbDf = pd.DataFrame(index=df.index, columns=range(0, num_topics))
    for i, doc in enumerate(docTopicProbMat):
        for topic in doc:
            docTopicProbDf.iloc[i, topic[0]] = topic[1]
    docTopicProbDf[new_col] = docTopicProbDf.idxmax(axis=1)
    df_topics = docTopicProbDf[new_col]
    # Merge with df
    df_new = pd.concat([df, df_topics], axis=1)
    return df_new


corpus.head()
corpus_description = corpus.astype(str)
processed_corpus_description = preprocess_text(corpus_description)

dictionary_description, doc_term_matrix_description = nlp_model_pipeline(processed_corpus_description)

ldamodel_description = LDA_topic_modelling(doc_term_matrix_description, dictionary_description, num_topics=4, passes=1)
p = pyLDAvis.gensim.prepare(ldamodel_description, doc_term_matrix_description, dictionary_description)
pyLDAvis.save_html(p, 'topic_4t.html')


###############################################################################
''' positive - negative feedback classifier'''
###############################################################################

#subset the train and test data 
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm

text_train_raw, text_test_raw,  label_train, label_test = train_test_split(text, label, 
                                                                  test_size = 0.2, random_state = 0 )
''' # build step by step
count_vect = CountVectorizer()
text_train = count_vect.fit_transform(text_train)
text_train.shape


tfidf_transformer = TfidfTransformer()
text_train_tfidf = tfidf_transformer.fit_transform(text_train)
text_train_tfidf.shape


clf = MultinomialNB().fit(text_train_tfidf, label_train)


predicted = clf.predict(text_train)

np.mean(predicted == label_train)

'''

#create pipline
#Naive Bayesian

text_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()),
])

text_clf = text_clf.fit(text_train_raw, label_train)
predicted = text_clf.predict(text_test_raw)
np.mean(predicted == label_test)


text_clf_sgd = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf-sgd', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, n_iter=5, random_state=42)),
 ])
 _ = text_clf_sgd.fit(text_train_raw, label_train)
 predicted_sgd = text_clf_sgd.predict(text_test_raw)
 np.mean(predicted_sgd == label_test)


text_clf_svm = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf-svm', svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=42, shrinking=True,
    tol=0.001, verbose=False)),
 ])
 _ = text_clf_svm.fit(text_train_raw, label_train)
 predicted_svm = text_clf_svm.predict(text_test_raw)
 np.mean(predicted_svm == label_test)



 from sklearn.model_selection import GridSearchCV
 parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-2, 1e-3),
 }

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)



###############################################################################
''' Word2Vector'''
###############################################################################

    processed_corpus = []
    english_words = set(nltk.corpus.words.words())
    english_stopwords = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer(r'[\w|!]+')
    for row in text:
        word_tokens = tokenizer.tokenize(row)
        word_tokens_lower = [t.lower() for t in word_tokens]
        word_tokens_lower_english = [t for t in word_tokens_lower if t in english_words or not t.isalpha()]
        word_tokens_no_stops = [t for t in word_tokens_lower_english if not t in english_stopwords]
        word_tokens_no_stops_lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in word_tokens_no_stops]
        processed_corpus.append(word_tokens_no_stops)

#build word2vec model
text_word2vec_model = word2vec.Word2Vec(processed_corpus, min_count=5, size=200)
#find the most similar words 
text_word2vec_model.most_similar(["amazing"])
#word to vectors of words 
text_word2vec_model.wv['spa'] 

from gensim.models.keyedvectors import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format('/tmp/vectors.txt', binary=False)  # C text format
word_vectors = KeyedVectors.load_word2vec_format('/tmp/vectors.bin', binary=True)  # C binary format

text_word2vec_model.wv.most_similar(positive=['spa', 'clinic'], negative=['massage'])

text_word2vec_model.wv.most_similar_cosmul(positive=['spa', 'clinic'], negative=['massage'])


text_word2vec_model.wv.doesnt_match("nail plastic Dr scar surgery".split())

text_word2vec_model.wv.similarity('plastic', 'surgery')



vocab = list(text_word2vec_model.wv.vocab)
X = text_word2vec_model[vocab]
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

fig = plt.figure(figsize = (18, 18))
ax = fig.add_subplot(1, 1, 1)
for word, pos in df.iloc[:50].iterrows():
    ax.annotate(word, pos)
ax.scatter(df['x'][:50], df['y'][:50])


# word frenquency
word_list = []

for i in range(len(processed_corpus)):
    word_list = word_list +  processed_corpus[i]
    
fdist = nltk.FreqDist(word_list)

for word, frequency in fdist.most_common(50):
    print(u'{};{}'.format(word, frequency))
    
#save model
text_word2vec_model.save("text_word2vec_gensim")




