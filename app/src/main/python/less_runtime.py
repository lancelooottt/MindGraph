from os.path import dirname, join

import pandas as pd
import numpy as np
from nltk.corpus import stopwords

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pickle
from textblob import Word
import re
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def output(textInput):
    lsvmModel = join(dirname(__file__),'trainedModel/lsvm_model.pkl')
    countvectModel = join(dirname(__file__),'trainedModel/count_vect_model.pkl')



    with open(lsvmModel, 'rb') as file:
        lsvm_model_try = pickle.load(file)

    with open(countvectModel, 'rb') as file:
        count_vect = pickle.load(file)

    tweets = pd.DataFrame([textInput])
    #tweets = pd.DataFrame(['I am very happy today! The atmosphere looks cheerful',
    #                       'Things are looking great. It was such a good day',
    #                       'Success is right around the corner. Lets celebrate this victory',
    #                       'Everything is more beautiful when you experience them with a smile!',
    #                       'Now this is my worst, okay? But I am gonna get better.',
    #                       'I am tired, boss. Tired of being on the road, lonely as a sparrow in the rain. I am tired of all the pain I feel',
    #                       'This is quite depressing. I am filled with sorrow',
    #                       'His death broke my heart. It was a sad day'])

    ##tweets = pd.DataFrame(["This is quite depressing. I am filled with sorrow"])

    # Doing some preprocessing on these tweets as done before
    tweets[0] = tweets[0].str.replace('[^\w\s]',' ')

    stop = stopwords.words('english')
    tweets[0] = tweets[0].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    #Lemmatisation
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    lemmatizer = WordNetLemmatizer()

    tweets[0] = tweets[0].apply(lambda x: " ".join([lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in x.split()]))

    # Extracting Count Vectors feature from our tweets
    tweet_count = count_vect.transform(tweets[0])

    #Predicting the emotion of the tweet using our already trained linear SVM
    tweet_pred = lsvm_model_try.predict(tweet_count)


    if tweet_pred == 0:
        emotion = "Anger"
    elif tweet_pred == 1:
        emotion = "Happiness"
    elif tweet_pred == 2:
        emotion = "Love"
    elif tweet_pred == 3:
        emotion = "Neutral"
    elif tweet_pred == 4:
        emotion = "Sadness"

    return "Predicted Emotion:", emotion
    ## result
    ## [0 0 0 0 1 1 1 1]