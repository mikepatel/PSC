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

    # ETL = Extraction, Transformation, Load
    d = Dataset()
    data = d.get_data()
    print(data)



