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

    # ETL = Extraction, Transformation, Load
    d = Dataset()
    df = d.lyrics_df

    # build list of lyrics from df
    lyrics_list = ["".join(i) for i in df.values]

    # convert list into one string (char list)
    lyrics = "".join(lyrics_list)
    print("Length of lyrics text: {}".format(len(lyrics)))
