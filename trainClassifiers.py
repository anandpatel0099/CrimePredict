import pickle
import os
import path

from getClassifiers import bernoulliNB,linearSVC,logisticRegression,multinomialNB,naiveBayes,nuSVC,SGDClassifier

load_featureset = open('pickled_data/featuresets.pickle', "rb")
featuresets = pickle.load(load_featureset)
load_featureset.close()

training_set = featuresets

#classifier                      = naiveBayes.getClassifier(path.naiveBayesPicklePATH,training_set)
#MultinomialNB_classifier        = multinomialNB.getClassifier(path.multinomialNBPicklePATH,training_set)
#BernoulliNB_classifier          = bernoulliNB.getClassifier(path.bernoulliNBPicklePATH,training_set)
LogisticRegression_classifier   = logisticRegression.getClassifier(path.logisticRegressionPicklePATH,training_set)
#LinearSVC_classifier            = linearSVC.getClassifier(path.linearSVCPicklePATH,training_set)
#SGDClassifier_classifier        = SGDClassifier.getClassifier(path.SGDClassifierPicklePATH,training_set)
#NuSVC_classifier                = nuSVC.getClassifier(path.NuSVCPicklePATH,training_set)

#testing_set = []
#print "Original NaiveBayes Accuracy : ",(nltk.classify.accuracy(classifier,testing_set)*100)
#print classifier.show_most_informative_features(15)
#print "MultinomialNB_classifier Accuracy : ",(nltk.classify.accuracy(MultinomialNB_classifier,testing_set)*100)
#print "BernoulliNB_classifier Accuracy : ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set) * 100)
#print "LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier,testing_set)) * 100
#print "LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100
#print "SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier,testing_set)) * 100
#print "NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier,testing_set)) * 100

