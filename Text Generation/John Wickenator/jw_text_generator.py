""""
John Jefferson III and Michael Patel
May 2019
Python 3.6.5
TF 1.12.0

Project Description:
    - Generate our own John Wick-esque script using RNNs!

Datasets:

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
MAX_SEQ_LENGTH = 500
BUFFER_SIZE = 10000
BATCH_SIZE = 64
EMBEDDING_DIM = 256
NUM_RNN_UNITS = 1024
NUM_EPOCHS = 50


################################################################################
# Data Preprocessing
class Dataset:
    def __init__(self):
        # read input from file
        file = os.path.join(os.getcwd(), "data\\johnwick.txt")
        _text = [line for line in open(file, encoding="utf-8")]
        _text = "".join(_text)

        # perform preprocessing
        self.text = self.preprocess(_text)

    # preprocess the text dataset
    @staticmethod
    def preprocess(text):
        # remove page numbers at bottom of page and at top right
        pattern = "Page.*[0-9].|[0-9].*"

        text = re.sub(
            pattern,
            "",
            text
        )

        return text

    # get data
    def get_data(self):
        return self.text


################################################################################
# Main
if __name__ == "__main__":
    # enable eager execution
    tf.enable_eager_execution()

    # print out TF version
    print("TF version: {}".format(tf.__version__))

    # ##### ETL ##### #
    # ETL = Extraction, Transformation, Load
    d = Dataset()
    data = d.get_data()

    # Tokenization: string => char tokens
    unique_chars = sorted(set(data))
    vocab_size = len(unique_chars)
    # print("Unique characters: {}".format(unique_chars))
    # print("Number of unique characters: {}".format(vocab_size))

    # Vectorization: convert char to vectors of int
    # create mapping from char to int
    char2idx = {u: i for i, u in enumerate(unique_chars)}

    # create mapping from int to char
    idx2char = {i: u for i, u in enumerate(unique_chars)}

    # create input and target sequences
    input_seqs = []
    target_seqs = []

    # build list of sequences of indices
    for i in range(0, len(data)-MAX_SEQ_LENGTH, MAX_SEQ_LENGTH):
        # create batches of char
        input_chars = data[i: i+MAX_SEQ_LENGTH]
        target_chars = data[i+1: i+1+MAX_SEQ_LENGTH]

        # convert each char in batch to int
        input_seqs.append([char2idx[i] for i in input_chars])
        target_seqs.append([char2idx[t] for t in target_chars])

    # shape: (n, MAX_SEQ_LENGTH) where n is the number of index sequences
    print("Shape of input sequences: {}".format(str(np.array(input_seqs).shape)))
    print("Shape of target sequences: {}".format(str(np.array(target_seqs).shape)))

    # use tf.data.Dataset to create batches and shuffle => TF model
    # (features, labels) === (input_seqs, target_seqs)
    sequences = tf.data.Dataset.from_tensor_slices((input_seqs, target_seqs))
    sequences = sequences.shuffle(buffer_size=BUFFER_SIZE)
    sequences = sequences.batch(batch_size=BATCH_SIZE, drop_remainder=True)

    print(sequences)

    # ##### MODEL ##### #
    m = build_model(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_rnn_units=NUM_RNN_UNITS,
        batch_size=BATCH_SIZE
    )
