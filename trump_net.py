""""
John Jefferson III and Michael Patel
September 2018
Python 3.6.5
TF 1.12.0

Project Description:
    - Text Generator modelling using RNNs

Dataset: Trump tweets from https://www.kaggle.com/kingburrito666/better-donald-trump-tweets/version/2

Notes:
    - using tf.keras and eager execution
    - character based model

"""
################################################################################
# Imports
import tensorflow as tf

import os
import re
import numpy as np
import pandas as pd


################################################################################
# Data Preprocessing
class Dataset:
    def __init__(self):
        # create csv after cleaning tweets dataset
        clean_csv = self._create_clean_csv()

        # create tweets dataframe object
        self.tweets_df = pd.read_csv(clean_csv)

    # perform preprocessing cleaning
    @staticmethod
    def _clean_tweets(tweet):
        # remove url links in tweet text
        tweet = re.sub(
            "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            tweet
        ).strip()

        return tweet

    # preprocess tweets and write output to csv
    def _create_clean_csv(self):
        dataset_csv = os.path.join(os.getcwd(), "DonaldTrumpTweetsDataset.csv")
        column_header = "Tweet_Text"
        tweets_df = pd.read_csv(dataset_csv, usecols=[column_header])
        _temp = []

        for index, row in tweets_df.iterrows():
            tweet = row[column_header]
            tweet = self._clean_tweets(tweet)
            _temp.append(tweet)

        # write cleaned tweets to a new csv
        dataset_clean_csv = os.path.join(os.getcwd(), "DJT_tweets_noURLs.csv")
        pd.DataFrame(_temp).to_csv(dataset_clean_csv, header=[column_header], index=None)
        return dataset_clean_csv

    # get dataframe of cleaned tweets
    def get_tweets_df(self):
        return self.tweets_df

    # get number of cleaned tweets
    def get_num_tweets(self):
        return len(self.tweets_df)


################################################################################
# Main
if __name__ == "__main__":
    # print out TF version
    print("\nTF version: {}".format(tf.__version__))

    #
    d = Dataset()
    tweets_df = d.get_tweets_df()
    tweets = []
    for index, row in tweets_df.iterrows():
        tweet = str(row[0])
        tweets.append(tweet)
    #tweets = tweets_df.values
    #print(tweets)
    num_tweets = d.get_num_tweets()
    print("Number of tweets: {}".format(num_tweets))
    #print(type(tweets))

    # text => tokens => vectors
    # segment text into char tokens
    unique_chars = set()
    for tweet in tweets:
        for char in tweet:
            unique_chars.add(char)


    # build a set of all unique characters from tweets
    """
    for index, row in tweets_df.iterrows():
        tweet = str(row["Tweet_Text"])
        for char in tweet:
            unique_chars.add(char)
    """
    unique_chars = sorted(unique_chars)
    #print(unique_chars)
    print("Number of Unique Chars: {}".format(len(unique_chars)))

    # create mapping from unique char -> indices
    char2idx = {u: i for i, u in enumerate(unique_chars)}
    print(char2idx)

    # create mapping from indices -> unique char
    idx2char = {i: u for i, u in enumerate(unique_chars)}
    print(idx2char)

    input_seq = []  # all char in chunk, except last
    target_seq = []  # all char in chunk, except first

    MAX_SENTENCE_LENGTH = 300

    # convert each char into int using char2idx
    for chunk in range(0, num_tweets-MAX_SENTENCE_LENGTH, MAX_SENTENCE_LENGTH):
        inputs = tweets[chunk: chunk+MAX_SENTENCE_LENGTH]
        targets = tweets[chunk+1: chunk+1+MAX_SENTENCE_LENGTH]

        input_seq.append([char2idx[i] for i in inputs])
        target_seq.append([char2idx[t] for t in targets])

    print("Shape of input sequence: {}".format(str(np.array(input_seq).shape)))
    print("Shape of target sequence: {}".format(str(np.array(target_seq).shape)))


