"""
John Jefferson III and Michael Patel
June 2019
Python 3.6.5
TF 1.12.0

Project Description:
    - Generate our own list of US names using RNNs!

Datasets:
    csv files of US names for decades: 1990s, 2000s, 2010s

Notes:

Things to examine:

"""
################################################################################
# Imports
import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

import tensorflow as tf


################################################################################
# Model hyperparameters
NUM_EPOCHS = 1
BATCH_SIZE = 64
MAX_SEQ_LENGTH = 20
EMBEDDING_DIM = 256
NUM_RNN_UNITS = 1024
BUFFER_SIZE = 10000
CHECKPOINT_PERIOD = NUM_EPOCHS  # how frequently to save checkpoints

# Generation parameters
START_STRING = "A"
NUM_CHAR_GEN = 8  # number of generated characters
TEMPERATURE = 0.8


################################################################################
# Data Preprocessing
class Dataset:
    def __init__(self):
        encoding = "windows-1252"

        # csv data files
        input_csvs = ["1990s.csv", "2000s.csv", "2010s.csv"]
        self.boys_df = pd.DataFrame()
        self.girls_df = pd.DataFrame()

        for csv in input_csvs:
            file = os.path.join(os.getcwd(), "data\\" + csv)
            b_temp_df = pd.read_csv(file, usecols=["Boy Name"], encoding=encoding)
            g_temp_df = pd.read_csv(file, usecols=["Girl Name"], encoding=encoding)
            self.boys_df = pd.concat([self.boys_df, b_temp_df])
            self.girls_df = pd.concat([self.girls_df, g_temp_df])


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

    # CLI Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--boys", help="Generate US boy names", action="store_true")
    parser.add_argument("--girls", help="Generate US girl names", action="store_true")
    args = parser.parse_args()

    if args.boys:
        df = d.boys_df

    elif args.girls:
        df = d.girls_df

    else:
        print("\nPlease provide an argument: ")
        parser.print_help()
        sys.exit(1)

    # ETL continued
    print("Size of Dataframe: {}".format(len(df)))

    # build list from df
    names_list = ["".join(i) for i in df.values]

    # convert list into one string (char list)
    names = "\n".join(names_list)

    # Tokenization: string => char tokens
    unique_chars = sorted(set(names))
    vocab_size = len(unique_chars)
    print("Number of unique characters: {}".format(vocab_size))

    # Vectorization: convert char to vectors of int
    # create mapping from char to int
    char2idx = {u: i for i, u in enumerate(unique_chars)}
    # print("Char2idx mappings: {}".format(char2idx))

    # create mapping from int to char
    idx2char = {i: u for i, u in enumerate(unique_chars)}
    # print("Idx2char mappings: {}".format(idx2char))

    # create input and target sequences
    input_seqs = []
    target_seqs = []

    # build list of sequences of indices
    for i in range(0, len(names)-MAX_SEQ_LENGTH, MAX_SEQ_LENGTH):
        # create batches of char
        input_chars = names[i: i+MAX_SEQ_LENGTH]
        target_chars = names[i+1: i+1+MAX_SEQ_LENGTH]

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
