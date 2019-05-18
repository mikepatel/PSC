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
    - https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/data/Dataset

"""
################################################################################
# Imports
import os
import re
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import Model


################################################################################
# Hyperparameters
MAX_SENTENCE_LENGTH = 300
BUFFER_SIZE = 10000
BATCH_SIZE = 256


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
    def _clean_tweet(tweet):
        # remove url links in tweet text
        tweet = re.sub(
            "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            tweet
        ).strip()

        # remove 'RT' in tweet text
        tweet = re.sub(
            "RT",
            "",
            tweet
        ).strip()

        return tweet

    # preprocess tweets and write output to csv
    def _create_clean_csv(self):
        dataset_csv = os.path.join(os.getcwd(), "DonaldTrumpTweetsDataset.csv")
        column_header = "Tweet_Text"
        in_df = pd.read_csv(dataset_csv, usecols=[column_header])
        _temp = []

        for index, row in in_df.iterrows():
            tweet = row[column_header]
            tweet = self._clean_tweet(tweet)
            _temp.append(tweet)

        # write cleaned tweets to a new csv
        dataset_clean_csv = os.path.join(os.getcwd(), "DJT_tweets_noURLs.csv")
        out_df = pd.DataFrame(_temp)
        out_df.replace("", np.nan, inplace=True)  # replace empty string cells with np.nan
        out_df = out_df.dropna()  # drop np.nan cells
        out_df.to_csv(dataset_clean_csv, header=[column_header], index=None)
        return dataset_clean_csv

    # get dataframe of cleaned tweets
    def get_tweets_df(self):
        return self.tweets_df

    # get number of cleaned tweets
    def get_num_tweets(self):
        return len(self.tweets_df)


################################################################################
# ML Model
class RNN(Model):
    def __init__(self):
        super(Model, self).__init__()


################################################################################
# Main
if __name__ == "__main__":
    # print out TF version
    print("\nTF version: {}".format(tf.__version__))

    ########################################
    # ETL = Extraction, Transformation, Load
    # get dataset
    d = Dataset()
    tweets_df = d.get_tweets_df()  # dataframe
    num_tweets = d.get_num_tweets()
    print("Number of tweets: {}".format(num_tweets))

    # build list of tweets from dataframe
    tweets = ["".join(i) for i in tweets_df.values]

    # convert tweet list to one long string since a string is a char list
    tweet_str = "".join(tweets)

    # text string => char tokens => vectors of int (1-dimensional arrays) => Model
    # segment text string into char tokens
    # then convert each char to int
    # feed arrays of int to the model
    unique_chars = sorted(set(tweet_str))  # a set is a collection of unique elements
    print("Number of unique chars: {}".format(len(unique_chars)))

    # create mapping from unique char -> indices
    char2idx = {u: i for i, u in enumerate(unique_chars)}
    # print(char2idx)

    # create mapping from indices -> unique char
    idx2char = {i: u for i, u in enumerate(unique_chars)}
    # print(idx2char)

    # list of sequences of indices
    input_seqs = []
    target_seqs = []

    # build lists of sequences of indices
    for i in range(0, len(tweet_str)-MAX_SENTENCE_LENGTH, MAX_SENTENCE_LENGTH):
        # create batches of char (i.e. list of char)
        inputs = tweet_str[i: i+MAX_SENTENCE_LENGTH]  # all char in chunk, except last
        targets = tweet_str[i+1: i+1+MAX_SENTENCE_LENGTH]  # all char in chunk, except first

        # convert each char in batch to int using char2idx
        input_seqs.append([char2idx[i] for i in inputs])
        target_seqs.append([char2idx[t] for t in targets])

    # shape: (x, MAX_SENTENCE_LENGTH) where x is number of index sequences
    print("Shape of input sequence: {}".format(str(np.array(input_seqs).shape)))
    print("Shape of target sequence: {}".format(str(np.array(target_seqs).shape)))

    # use tf.data.Dataset to create batches and shuffle => TF Model
    input_data = tf.data.Dataset.from_tensor_slices((input_seqs, target_seqs))
    input_data = input_data.shuffle(buffer_size=BUFFER_SIZE)
    input_data = input_data.batch(batch_size=BATCH_SIZE, drop_remainder=True)

    ########################################
    # Model
