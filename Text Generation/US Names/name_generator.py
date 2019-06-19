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


# Generation parameters


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
    print(names)
