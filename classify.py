from __future__ import division
import warnings
warnings.filterwarnings("ignore")
import trainClassifiers as tc
from nltk.classify import ClassifierI
from nltk.tokenize import TweetTokenizer
import re
import pickle
import nltk

from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english'))

tknzr = TweetTokenizer()

load_word_features = open('pickled_data/word_features.pickle', "rb")
word_features = pickle.load(load_word_features)
load_word_features.close()

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

def processTweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    tweet = re.sub(r'@[^\s]+','AT_USER',tweet)
    tweet = re.sub(r'[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = tweet.strip('\'"')
    return tweet

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes=[]
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return max(set(votes),key=votes.count)

    def confidence(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(max(set(votes),key=votes.count))
        conf = float(choice_votes / len(votes))
        return conf

voted_classifier = VoteClassifier(tc.LogisticRegression_classifier)
                                    #tc.classifier,
                                  #tc.MultinomialNB_classifier,
                                  #tc.BernoulliNB_classifier,
                                  #tc.LogisticRegression_classifier,
                                  #tc.SGDClassifier_classifier,
                                  #tc.LinearSVC_classifier,
                                  #tc.NuSVC_classifier)

#print "voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100
def relatibility(text):
    feats = find_features([word.lower() for word in tknzr.tokenize(processTweet(text)) if word not in stop_words])
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)
    