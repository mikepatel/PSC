""""
John Jefferson III and Michael Patel
September 2018
Python 3.6.5
TF 1.12.0

Project Description:
    - Text Generator modelling using RNNs
    - Predict the next character in a sequence

Dataset: Trump tweets from https://www.kaggle.com/kingburrito666/better-donald-trump-tweets/version/2

Notes:
    - using tf.keras and eager execution
    - character based RNN model
    - https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/data/Dataset
    - https://www.tensorflow.org/tutorials/sequences/text_generation

"""
################################################################################
# Imports
import os
import re
import numpy as np
import pandas as pd

import tensorflow as tf


################################################################################
# Model hyperparameters
MAX_SEQ_LENGTH = 300
BUFFER_SIZE = 10000
BATCH_SIZE = 64
VOCAB_SIZE = 0  # redefined later in code
EMBEDDING_DIM = 256
NUM_RNN_UNITS = 1024


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
class Model(tf.keras.Model):
    # implement model layers
    def __init__(self, vocab_size, embedding_dim, num_rnn_units):
        super(Model, self).__init__()

        # Layer 1: Embedding
        # An embedding transforms positive ints (indices) into dense vectors of floats
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,  # max int number + 1
            output_dim=embedding_dim  # dimension of dense embedding
        )

        # Layer 2: GRU
        # check for GPU
        if tf.test.is_gpu_available():
            self.gru = tf.keras.layers.CuDNNGRU(
                units=num_rnn_units,  # dimensionality of output space
                return_sequences=True,  # return full sequence
                return_state=True  # return last state and output
            )
        else:
            self.gru = tf.keras.layers.GRU(
                units=num_rnn_units,
                return_sequences=True,
                return_state=True
            )

        # Layer 3: Fully Connected
        self.fc = tf.keras.layers.Dense(vocab_size)

    # implement forward pass
    def call(self, inputs):
        inputs = self.embedding(inputs)

        h, states = self.gru(inputs)
        h = tf.reshape(h, (-1, h.shape[2]))

        outputs = self.fc(h)

        return outputs, states


################################################################################
# Main
if __name__ == "__main__":
    tf.enable_eager_execution()

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
    print("Length of text: {}".format(len(tweet_str)))

    # text string => char tokens => vectors of int (1-dimensional arrays) => Model
    # segment text string into char tokens
    # then convert each char to int
    # feed arrays of int to the model
    unique_chars = sorted(set(tweet_str))  # a set is a collection of unique elements
    print("Number of unique chars: {}".format(len(unique_chars)))
    VOCAB_SIZE = len(unique_chars)

    # create mapping from unique char -> indices
    char2idx = {u: i for i, u in enumerate(unique_chars)}

    # create mapping from indices -> unique char
    idx2char = {i: u for i, u in enumerate(unique_chars)}

    # list of sequences of indices
    input_seqs = []
    target_seqs = []

    # build lists of sequences of indices
    for i in range(0, len(tweet_str)-MAX_SEQ_LENGTH, MAX_SEQ_LENGTH):
        # create batches of char (i.e. list of char)
        inputs = tweet_str[i: i+MAX_SEQ_LENGTH]  # all char in chunk, except last
        targets = tweet_str[i+1: i+1+MAX_SEQ_LENGTH]  # all char in chunk, except first

        # convert each char in batch to int using char2idx
        input_seqs.append([char2idx[i] for i in inputs])  # as int
        target_seqs.append([char2idx[t] for t in targets])  # as int

    # shape: (x, MAX_SENTENCE_LENGTH) where x is number of index sequences
    print("Shape of input sequence: {}".format(str(np.array(input_seqs).shape)))
    print("Shape of target sequence: {}".format(str(np.array(target_seqs).shape)))

    # use tf.data.Dataset to create batches and shuffle => TF Model
    # features => input_seqs
    # labels => target_seqs
    sequences = tf.data.Dataset.from_tensor_slices((input_seqs, target_seqs))
    sequences = sequences.shuffle(buffer_size=BUFFER_SIZE)
    sequences = sequences.batch(batch_size=BATCH_SIZE, drop_remainder=True)

    print(sequences)

    ########################################
    # Model
    m = Model(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_rnn_units=NUM_RNN_UNITS
    )
