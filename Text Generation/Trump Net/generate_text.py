"""
Michael Patel
June 2020

Project description:
    Trump tweet generator

File description:
    Test script to generate text
"""
################################################################################
# Imports
import os
import tensorflow as tf

from trump_net import build_model


################################################################################
# Main
if __name__ == "__main__":
    model = build_model(

    )

    # load weights
    save_filepath = os.path.join(os.getcwd(), "saved_model")
    model.load_weights(save_filepath)
