""""
John Jefferson III and Michael Patel
September 2018 -
Python 3.6.5
TF

Project Description:
    - Text Generator modelling using RNNs

Dataset: Trump tweets from https://www.kaggle.com/kingburrito666/better-donald-trump-tweets/version/2

Notes:
    - using tf.keras and eager execution
    - character based model

"""
################################################################################
# IMPORTs
import tensorflow as tf

import os
import csv


################################################################################
# Data Preprocessing

# build tweets list
def build_tweets_list():
    tweets_data_csv = os.path.join(os.getcwd(), "DonaldTrumpTweetsDataset_JustTweetText.csv")
    tweets = []
    with open(tweets_data_csv, newline="", encoding="utf8") as f:
        csv_reader = csv.reader(f, delimiter=",")

        line_count = 0

        for row in csv_reader:
            if line_count == 0:  # column names
                line_count += 1
            else:
                tweets.append(str(row[0]))  # first column in csv
                line_count += 1

    print(len(tweets))
    return tweets


################################################################################
# Main
if __name__ == "__main__":
    print("\nTF version: {}".format(tf.__version__))
    TWEETS = build_tweets_list()
