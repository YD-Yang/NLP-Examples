
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.classify import decisiontree
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import twitter_samples

twitter_samples.fileids()


#read the movie review data set
def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict
    
    
neg_reviews = []
for fileid in movie_reviews.fileids('neg'):    
    words = movie_reviews.words(fileid)
    neg_reviews.append((create_word_features(words), "negative"))


print(neg_reviews[0])    
print(len(neg_reviews))

pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append((create_word_features(words), "positive"))
    
#print(pos_reviews[0])    
print(len(pos_reviews))

 
random.shuffle(pos_reviews)
random.shuffle(neg_reviews)
reviews =  neg_reviews + pos_reviews
train_set = neg_reviews[:750] + pos_reviews[:750]
test_set =  neg_reviews[750:] + pos_reviews[750:]
print(len(train_set),  len(test_set))
 
 
 
#train NaiveBayes Classifier 
NB_classifier = NaiveBayesClassifier.train(train_set) 

accuracy = nltk.classify.util.accuracy(NB_classifier, test_set)
print(accuracy * 100)

#an example
review_test = '''You are Awesome!!   Keep Smiling & have an amazing holiday season <ed><U+00A0><U+00BD><ed><U+00B8><U+008A>'''

review_test = '''On my third ...awesome car...exciting to drive...fast, peppy, and getting 47 MPG average.'''

words = word_tokenize(review_test)
words = create_word_features(words)
NB_classifier.classify(words)



#train decision tree Classifier 
sample = neg_reviews[:10] + pos_reviews[:10]

classifier =nltk.DecisionTreeClassifier.train(train_set, entropy_cutoff=0.2)

accuracy = nltk.classify.util.accuracy(classifier, train_set)
print(accuracy * 100)

#an example
review_test = '''You are Awesome!!   Keep Smiling & have an amazing holiday season <ed><U+00A0><U+00BD><ed><U+00B8><U+008A>'''

review_test = '''On my third ....awesome car...exciting to drive...fast, peppy, and getting 47 MPG average.'''

words = word_tokenize(review_test)
words = create_word_features(words)
classifier.classify(words)


