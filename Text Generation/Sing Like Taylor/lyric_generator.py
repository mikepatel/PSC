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
NUM_CHAR_GEN = MAX_SEQ_LENGTH  # number of generated characters
CHECKPOINT_PERIOD = NUM_EPOCHS  # how frequently to save checkpoints


################################################################################
# Data Preprocessing
class Dataset: