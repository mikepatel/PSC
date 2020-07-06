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


################################################################################
# Main
if __name__ == "__main__":
    # load model
    save_filepath = os.path.join(os.getcwd(), "saved_model")
    model = tf.keras.models.load_model(save_filepath)
