""""
John Jefferson III and Michael Patel
June 2019
Python 3.6.5
TF 1.12.0

Project Description:
    - Generate our own Taylor Swift music lyrics using RNNs!

Datasets:
    csv file of Taylor Swift lyrics

Notes:

Things to examine:

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
MAX_SEQ_LENGTH = 40
BUFFER_SIZE = 10000
BATCH_SIZE = 64
EMBEDDING_DIM = 256
NUM_RNN_UNITS = 2048
NUM_EPOCHS = 50
NUM_CHAR_GEN = 40  # number of generated characters
CHECKPOINT_PERIOD = NUM_EPOCHS  # how frequently to save checkpoints


################################################################################
# Data Preprocessing
class Dataset:
    def __init__(self):
        self.encoding = "windows-1252"

        # csv dataset file
        input_csv = "taylor_swift_lyrics.csv"
        column_header = "lyric"
        input_file = os.path.join(os.getcwd(), "data\\" + input_csv)

        # lyrics df
        self.lyrics_df = pd.read_csv(input_file, usecols=[column_header], encoding=self.encoding)


################################################################################
# Main
if __name__ == "__main__":
    # enable eager execution
    tf.enable_eager_execution()

    # print out TF version
    print("TF version: {}".format(tf.__version__))

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    d = Dataset()
    df = d.lyrics_df

    # build list of lyrics from df
    lyrics_list = ["".join(i) for i in df.values]

    # convert list into one string (char list)
    lyrics = "".join(lyrics_list)
    # print("Length of lyrics text: {}".format(len(lyrics)))

    # Tokenization: string => char tokens
    unique_chars = sorted(set(lyrics))
    vocab_size = len(unique_chars)
    # print("Unique characters: {}".format(unique_chars))
    print("Number of unique characters: {}".format(vocab_size))

    # Vectorization: convert char to vectors of int
    # create mapping from char to int
    char2idx = {u: i for i, u in enumerate(unique_chars)}

    # create mapping from int to char
    idx2char = {i: u for i, u in enumerate(unique_chars)}

    # create input and target sequences
    input_seqs = []
    target_seqs = []

    # build list of sequences of indices
    for i in range(0, len(lyrics)-MAX_SEQ_LENGTH, MAX_SEQ_LENGTH):
        # create batches of char
        input_chars = lyrics[i: i+MAX_SEQ_LENGTH]
        target_chars = lyrics[i+1: i+1+MAX_SEQ_LENGTH]

        # convert each char in batch to int
        input_seqs.append([char2idx[i] for i in input_chars])
        target_seqs.append([char2idx[t] for t in target_chars])

    # shape: (n, MAX_SEQ_LENGTH) where n is the number of index sequences
    print("Shape of input sequences: {}".format(str(np.array(input_seqs).shape)))
    print("Shape of target sequences: {}".format(str(np.array(target_seqs).shape)))

    # use tf.data.Dataset to create batches and shuffle => TF model
    # (features, labels) == (input_seqs, target_seqs)
    sequences = tf.data.Dataset.from_tensor_slices((input_seqs, target_seqs))
    sequences = sequences.shuffle(buffer_size=BUFFER_SIZE)
    sequences = sequences.batch(batch_size=BATCH_SIZE, drop_remainder=True)

    print("Shape of batches: {}".format(sequences))

    # ----- MODEL ----- #
