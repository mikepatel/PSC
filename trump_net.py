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
import csv
import re


################################################################################
# Data Preprocessing
class Dataset:
    def __init__(self):
        # csv files
        self.dataset_csv = os.path.join(os.getcwd(), "DonaldTrumpTweetsDataset_JustTweetText.csv")
        self.dataset_clean_csv = os.path.join(os.getcwd(), "DJT_tweets_noURLs.csv")

        self.tweets = []

        # check if csv of cleaned tweets already exists
        if not os.path.exists(self.dataset_clean_csv):
            self.build_tweets_csv()

        self.build_tweets_list()

    # preprocess tweets and write output to csv
    def build_tweets_csv(self):
        csv_reader = csv.reader(open(self.dataset_csv, mode="r", encoding="utf8"))
        csv_writer = csv.writer(open(self.dataset_clean_csv, mode="w", newline="", encoding="utf8"))

        line_count = 0

        for row in csv_reader:
            if line_count == 0:  # column names
                line_count += 1
            else:
                tweet = str(row[0]).rstrip()
                tweet = str(re.sub(
                    "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                    "",
                    tweet
                )).rstrip()
                csv_writer.writerow([tweet])
                line_count += 1

    # build list of cleaned tweets
    def build_tweets_list(self):
        csv_reader = csv.reader(open(self.dataset_clean_csv, mode="r", encoding="utf8"))

        for row in csv_reader:
            tweet = str(row[0]).strip()
            self.tweets.append(tweet)

    # get list of cleaned tweets
    def get_tweets(self):
        return self.tweets

    # get number of cleaned tweets
    def get_num_tweets(self):
        return len(self.tweets)


################################################################################
# Model


################################################################################
# Main
if __name__ == "__main__":
    print("\nTF version: {}".format(tf.__version__))

    #
    d = Dataset()
    tweets = d.tweets
    print("Number of tweets: {}".format(d.get_num_tweets()))

    # build a set of all unique characters from tweets
    unique_chars = set()
    for tweet in tweets:
        for char in tweet:
            unique_chars.add(char)
    unique_chars = sorted(unique_chars)
    print(unique_chars)
    print(len(unique_chars))

    # create mapping from unique char -> indices
    char2idx = {u: i for i, u in enumerate(unique_chars)}
    print(char2idx)

    # create mapping from indices -> unique char
    idx2char = {i: u for i, u in enumerate(unique_chars)}
    print(idx2char)
