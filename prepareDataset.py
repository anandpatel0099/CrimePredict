import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import re
import pickle
from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))

tknzr = TweetTokenizer()

def processTweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    tweet = re.sub(r'@[^\s]+','AT_USER',tweet)
    tweet = re.sub(r'[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = tweet.strip('\'"')
    return tweet

dataset = open("dataset/labeled_tweets.txt","r").read()
documents = []
dataset_tweets = []

for r in dataset.split('\n'):
    processedTweet = []
    for doc_word in tknzr.tokenize(processTweet(r[2:])):
        processedTweet.append(doc_word.lower())
    if(len(r)>0 and r[0]=='1'):
        documents.append((processedTweet, "related"))
    elif(len(r)>0 and r[0]=='0'):
        documents.append((processedTweet, "unrelated"))
    dataset_tweets.append(processedTweet)

save_documents = open("pickled_data/documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = []
for dataset_tweet in dataset_tweets:
    for word in dataset_tweet:
        if word not in stop_words:
            all_words.append(word.lower())

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())

save_word_features = open("pickled_data/word_features.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev),category) for (rev,category) in documents]

save_featuresets = open("pickled_data/featuresets.pickle", "wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()