# crime_classifier

>>>dataset/labeled_tweets.txt 
- It's a primary labelled dataset used to train the model

>>>prepareDataset.py
- Prepares word_features : extracts words based on the word frequency
- Prepares featuresets: a structured dataset, ready to train the classifiers with

>>>pickled_data/word_features
>>>pickled_data/featuresets
- saved word_features and featuresets python object as a file

>>>path.py
- stores the metadata of directory and file paths


>>>getClassifiers/logisticRegression.py
-  used to import pickled classifiers from pickled_classifiers directory or train and create classifiers if pickles are not available

>>>pickled_classifiers/*
- saved classifiers' python objects as files

>>>trainClassifiers.py
- it loads/trains the classifiers from getClassifiers directory

>>>classify.py
- It uses classifiers loaded by trainClassifiers
- It has methods to pre-process tweets.
- voteClassifer class - which is used to classify tweets based on the votes(predicted result) by each loaded classifiers.( I tried to do   this but its not working)
- relatibility(tweet) - Final method to process tweet and fetch the result. 

>>>TEST.py
- a few examples of tweets and showing output of classification obtained by relatibility(tweet). 

>>>INPUT
- Type any sentence in TEST.py to check wether it is related to crime or not.

>>>OUTPUT
- run TEST.py, if it shows  "('related', 1.0)" for any sentence then it is related to crime otherwise it will show 
  "('unrelated', 1.0)".


