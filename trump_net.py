""""
John Jefferson III and Michael Patel
September 2018
Python 3.6.5
TF 1.12.0

Project Description:
    - Text Generator modelling using RNNs
    - Predict the next character in a sequence

Datasets: Trump tweets from
    - https://www.kaggle.com/kingburrito666/better-donald-trump-tweets/version/2 (~7k)
    - http://www.trumptwitterarchive.com/archive (~35k)

Notes:
    - using tf.keras and eager execution
    - character based RNN model
    - https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/data/Dataset
    - https://www.tensorflow.org/tutorials/sequences/text_generation

Things to examine:
    - compare GRU vs LSTM
    - number of layers for GRU/LSTM: 1, 2, 3
    - compare character-based vs word-based model
    - different data sets
    - dropout:

"""
################################################################################
# Imports
import os
import re
import numpy as np
import pandas as pd
from datetime import datetime

import tensorflow as tf


################################################################################
# Model hyperparameters
MAX_SEQ_LENGTH = 300
NUM_GENERATE = 280  # tweet length is 280 characters
BUFFER_SIZE = 10000
BATCH_SIZE = 128
VOCAB_SIZE = 0  # redefined later in code
EMBEDDING_DIM = 256
NUM_RNN_UNITS = 1024
NUM_EPOCHS = 50


################################################################################
# Data Preprocessing
class Dataset:
    def __init__(self):
        self.encoding = "windows-1252"

        # create csv after cleaning tweets dataset
        clean_csv = self._create_clean_csv()

        # create tweets dataframe object
        self.tweets_df = pd.read_csv(clean_csv, encoding=self.encoding)

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

        # remove 'amp;' in tweet text
        tweet = re.sub(
            "amp;",
            "",
            tweet
        ).strip()

        return tweet

    # preprocess tweets and write output to csv
    def _create_clean_csv(self):
        input_csv = "DonaldTrumpTweetsDataset.csv"
        input_csv = "Trump_Twitter_Archive.csv"

        column_header = "Tweet_Text"

        dataset_csv = os.path.join(os.getcwd(), input_csv)
        in_df = pd.read_csv(dataset_csv, usecols=[column_header], encoding=self.encoding)
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
        out_df.to_csv(dataset_clean_csv, header=[column_header], index=None, encoding=self.encoding)
        return dataset_clean_csv

    # get dataframe of cleaned tweets
    def get_tweets_df(self):
        return self.tweets_df

    # get number of cleaned tweets
    def get_num_tweets(self):
        return len(self.tweets_df)


################################################################################
# RNN
def build_model(vocab_size, embedding_dim, num_rnn_units, batch_size):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        batch_size=batch_size
    ))

    if tf.test.is_gpu_available():
        model.add(tf.keras.layers.CuDNNGRU(
            units=num_rnn_units,
            return_sequences=True,
            stateful=True
        ))
    else:
        model.add(tf.keras.layers.GRU(
            units=num_rnn_units,
            return_sequences=True,
            stateful=True
        ))

    model.add(tf.keras.layers.Dense(
        units=vocab_size
    ))

    return model


################################################################################
# generate text
def generate(model, start_char):
    input_eval = [char2idx[s] for s in start_char]
    input_eval = tf.expand_dims(input_eval, 0)

    text_gen = []

    temperature = 0.8

    model.reset_states()

    for i in range(NUM_GENERATE):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predictions_id = tf.multinomial(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predictions_id], 0)

        text_gen.append(idx2char[predictions_id])

    return start_char + "".join(text_gen)


################################################################################
# Main
if __name__ == "__main__":
    # enable eager execution
    tf.enable_eager_execution()

    # print out TF version
    print("TF version: {}".format(tf.__version__))

    ########################################
    # ETL = Extraction, Transformation, Load
    # get dataset
    d = Dataset()
    tweets_df = d.get_tweets_df()  # dataframe
    num_tweets = d.get_num_tweets()
    print("Number of tweets: {}".format(num_tweets))

    quit()

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
    m = build_model(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_rnn_units=NUM_RNN_UNITS,
        batch_size=BATCH_SIZE
    )

    m.summary()

    # loss function
    def loss_fn(labels, logits):
        return tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=logits
        )

    m.compile(
        loss=loss_fn,
        optimizer=tf.train.AdamOptimizer()
    )

    # callbacks for checkpoints, TensorBoard
    checkpoint_dir = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    print(checkpoint_dir)
    history_file = os.path.join(checkpoint_dir, "checkpoint_{epoch}")
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=history_file,
        save_weights_only=True,
        verbose=1
    )
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=checkpoint_dir)

    # train model
    history = m.fit(
        x=sequences.repeat(),
        epochs=NUM_EPOCHS,
        callbacks=[save_callback, tb_callback],
        steps_per_epoch=len(tweet_str)//MAX_SEQ_LENGTH//BATCH_SIZE,
        verbose=1
    )

    ########################################
    # Generate Output
    # run model with different batch size, so need to rebuild model
    m = build_model(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_rnn_units=NUM_RNN_UNITS,
        batch_size=1  #
    )
    m.load_weights(tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir))
    m.build(tf.TensorShape([1, None]))
    m.summary()

    gen_tweet = generate(model=m, start_char="M")

    print("\n################################################################################")
    print("GENERATED TWEET: ")
    print("\"" + gen_tweet + "\"")
    print("################################################################################")
