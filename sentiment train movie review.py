
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

review_test = '''On my third hybrid 2017 Honda Accord....awesome car...exciting to drive...fast, peppy, and getting 47 MPG average.'''

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

review_test = '''On my third hybrid 2017 Honda Accord....awesome car...exciting to drive...fast, peppy, and getting 47 MPG average.'''

words = word_tokenize(review_test)
words = create_word_features(words)
classifier.classify(words)


###############################
# read honda review data 
import csv
import pandas as pd
   
with open("post_comments.csv", "rb") as f:
    reader = csv.reader(f)
    i = reader.next()
    comments = [row for row in reader]
        
honda_comments = [comments[i][2] for i in range(shape(comments)[0])]
pred_atti = []
for i in range(len(honda_comments)):
    review_test = honda_comments[i]
    words = word_tokenize(review_test.decode('latin-1'))
    words = create_word_features(words)
    pred_atti.append( classifier.classify(words))


NB_pred = 'NB_pred.csv'
with open(NB_pred, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in pred_atti:
        writer.writerow([val])    


Tree_pred = 'Tree_pred.csv'
with open(Tree_pred, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in pred_atti:
        writer.writerow([val])    
    

##############################################################
# save and load file    
import pickle
f = open('tree_classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()

f = open('tree_classifier.pickle', 'rb')
tree_classifier = pickle.load(f)
f.close()


f = open('NB_classifier.pickle', 'wb')
pickle.dump(NB_classifier, f)
f.close()

f = open('NB_classifier.pickle', 'rb')
Nb = pickle.load(f)
f.close()
